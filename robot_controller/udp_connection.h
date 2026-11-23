#pragma once

#include <arpa/inet.h>
#include <endian.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "types.h"
#include "edt.h"

constexpr int MY_PORT   = 5001;  // C++ listens here
constexpr int PEER_PORT = 5002;  // Python listens here

// request header: seq(u32), x, y, theta, sensor_radius, goal_x, goal_y (f64), rows, cols (u32), is_final (u8)
// is_final: 1 = stop on this goal (Python sends 1 for the last one), read but unused by this controller
// followed by the sensor window mask (rows*cols u8, indexed [x][y], 1 = free, 0 = obstacle/unknown/outside the sensor
// circle), centred on the robot cell (round half to even), rows = cols = 2 * ceil(sensor_radius) + 1,
// then the route nodes: count (u32) + count * NODE_SIZE
// the EDT is computed in C++ and sent back before the reply for display (edt.h)
constexpr size_t REQ_HEADER_SIZE = 4 + 6 * 8 + 2 * 4 + 1;
// route node: centre x, y, axis1 x, y, half width, half height (f64)
constexpr size_t NODE_SIZE = 6 * 8;
// reply: seq(u32), v, w (f64)
// then for display: field count (u32) + count * (x, y, vx, vy) (f32), kernel count (u32) + count * (centre x, y, mean x, y) (f32)
constexpr size_t REPLY_SIZE = 4 + 2 * 8;
// largest UDP payload, the field is cut to fit
constexpr size_t MAX_REPLY_SIZE = 65507;
// sent by Python once the final goal is reached, echoed back before shutting down
constexpr char STOP_MSG[] = {'S', 'T', 'O', 'P'};
// EDT window (edt.h), sent before the reply and split into chunks because a float EDT of a large window (the car's
// 181 x 181) does not fit in one datagram; every chunk (big-endian):
//   magic "EDTW", seq (u32), chunk index, chunk count (u16), first cell, cell count (u32),
//   origin x, y (f64, centre of cell [0][0]), size x, size y (u32),
//   max distance, robot x, y, theta, robot radius, footprint front, rear, half width, footprint clearance,
//   closest x, y (f64)
//   then cell count distances (f32, indexed [i * size_y + j], 0 = blocked)
// all lengths in map cells (px); a circular robot has footprint front = rear = half width = 0 and its clearance
// is the EDT at its centre minus its radius, closest = its centre
constexpr char EDT_MAGIC[] = {'E', 'D', 'T', 'W'};
constexpr size_t EDT_HEADER_SIZE = 4 + 4 + 2 * 2 + 2 * 4 + 2 * 8 + 2 * 4 + 11 * 8;
// cells per chunk, so a chunk fits in the largest UDP payload
constexpr size_t EDT_CHUNK_CELLS = (MAX_REPLY_SIZE - EDT_HEADER_SIZE) / 4;

// big-endian (network byte order) helpers
inline uint32_t get_u32(const uint8_t* p) { uint32_t t; std::memcpy(&t, p, 4); return ntohl(t); }
inline float    get_f32(const uint8_t* p) { uint32_t t = get_u32(p); float f;  std::memcpy(&f, &t, 4); return f; }
inline double   get_f64(const uint8_t* p) { uint64_t t; std::memcpy(&t, p, 8); t = be64toh(t); double d; std::memcpy(&d, &t, 8); return d; }

inline void put_u16(uint8_t* p, uint16_t v) { v = htons(v); std::memcpy(p, &v, 2); }
inline void put_u32(uint8_t* p, uint32_t v) { v = htonl(v); std::memcpy(p, &v, 4); }
inline void put_f32(uint8_t* p, float f) { uint32_t t; std::memcpy(&t, &f, 4); put_u32(p, t); }
inline void put_f64(uint8_t* p, double d) { uint64_t t; std::memcpy(&t, &d, 8); t = htobe64(t); std::memcpy(p, &t, 8); }

struct Request {
    uint32_t seq;
    RobotState pose;
    double sensor_radius;
    uint32_t rows, cols;
    RobotState goal;  // current temp goal (x, y), theta unused
    bool is_final = false;  // stop on the goal (unused by this controller)
    std::vector<unsigned char> isFree;  // rows*cols, row-major [x][y], 1 = free
    std::vector<float> edt;    // rows*cols, row-major [x][y], distance to nearest obstacle/unknown (0 = blocked), filled from isFree in C++
    std::vector<RectRegion> nodes;  // current route node first, then the next ones up to the temp goal (can be empty)
};

inline bool parse_request(const uint8_t* buf, size_t n, Request& req) {
    if (n < REQ_HEADER_SIZE) return false;

    req.seq           = get_u32(buf);
    req.pose.x        = get_f64(buf + 4);
    req.pose.y        = get_f64(buf + 12);
    req.pose.theta    = get_f64(buf + 20);
    req.sensor_radius = get_f64(buf + 28);
    req.goal.x        = get_f64(buf + 36);
    req.goal.y        = get_f64(buf + 44);
    req.rows          = get_u32(buf + 52);
    req.cols          = get_u32(buf + 56);
    req.is_final      = buf[60] != 0;

    size_t cells = static_cast<size_t>(req.rows) * req.cols;
    size_t nodes_offset = REQ_HEADER_SIZE + cells;
    if (n < nodes_offset + 4) return false;

    const uint8_t* mask_p = buf + REQ_HEADER_SIZE;

    req.isFree.resize(cells);
    for (size_t i = 0; i < cells; ++i) req.isFree[i] = mask_p[i] != 0;

    size_t node_count = get_u32(buf + nodes_offset);
    if (n != nodes_offset + 4 + node_count * NODE_SIZE) return false;

    const uint8_t* node_p = buf + nodes_offset + 4;

    req.nodes.resize(node_count);
    for (size_t k = 0; k < node_count; ++k, node_p += NODE_SIZE) {
        RectRegion& node = req.nodes[k];
        node.center.x   = get_f64(node_p);
        node.center.y   = get_f64(node_p + 8);
        node.axis1.x    = get_f64(node_p + 16);
        node.axis1.y    = get_f64(node_p + 24);
        node.halfWidth  = get_f64(node_p + 32);
        node.halfHeight = get_f64(node_p + 40);
    }

    return true;
}

// reply: the command first (same 20 bytes as before), then the field and the kernels for display
inline std::vector<uint8_t> build_reply(uint32_t seq, const ControlInput& u, const std::vector<float>& field, const std::vector<NavKernel>& kernels) {
    const size_t kernelBytes = 4 + kernels.size() * 4 * 4;
    const size_t maxPoints = (MAX_REPLY_SIZE - REPLY_SIZE - 4 - kernelBytes) / (4 * 4);
    const size_t points = std::min(field.size() / 4, maxPoints);

    std::vector<uint8_t> out(REPLY_SIZE + 4 + points * 4 * 4 + kernelBytes);
    uint8_t* p = out.data();

    put_u32(p, seq);
    put_f64(p + 4, u.v);
    put_f64(p + 12, u.omega);
    p += REPLY_SIZE;

    put_u32(p, static_cast<uint32_t>(points));
    p += 4;
    for (size_t i = 0; i < points * 4; ++i, p += 4) {
        put_f32(p, field[i]);
    }

    put_u32(p, static_cast<uint32_t>(kernels.size()));
    p += 4;
    for (const NavKernel& k : kernels) {
        put_f32(p, static_cast<float>(k.center.x));
        put_f32(p + 4, static_cast<float>(k.center.y));
        put_f32(p + 8, static_cast<float>(k.mu.x));
        put_f32(p + 12, static_cast<float>(k.mu.y));
        p += 16;
    }

    return out;
}

// the EDT window split into datagrams, see EDT_HEADER_SIZE above
inline std::vector<std::vector<uint8_t>> build_edt_packets(uint32_t seq, const EdtWindowInfo& info, const std::vector<float>& distances) {
    const size_t cells = distances.size();
    const size_t chunks = std::max<size_t>(1, (cells + EDT_CHUNK_CELLS - 1) / EDT_CHUNK_CELLS);

    std::vector<std::vector<uint8_t>> packets;
    packets.reserve(chunks);

    for (size_t c = 0; c < chunks; ++c) {
        const size_t first = c * EDT_CHUNK_CELLS;
        const size_t count = std::min(EDT_CHUNK_CELLS, cells - first);

        std::vector<uint8_t> out(EDT_HEADER_SIZE + count * 4);
        uint8_t* p = out.data();

        std::memcpy(p, EDT_MAGIC, 4);
        put_u32(p + 4, seq);
        put_u16(p + 8, static_cast<uint16_t>(c));
        put_u16(p + 10, static_cast<uint16_t>(chunks));
        put_u32(p + 12, static_cast<uint32_t>(first));
        put_u32(p + 16, static_cast<uint32_t>(count));
        put_f64(p + 20, info.originX);
        put_f64(p + 28, info.originY);
        put_u32(p + 36, info.sizeX);
        put_u32(p + 40, info.sizeY);

        const double values[11] = {info.maxDistance, info.robotX, info.robotY, info.robotTheta, info.robotRadius,
                                   info.footprintFront, info.footprintRear, info.footprintHalfWidth,
                                   info.footprintClearance, info.closestX, info.closestY};
        for (int k = 0; k < 11; ++k) {
            put_f64(p + 44 + 8 * k, values[k]);
        }

        p += EDT_HEADER_SIZE;
        for (size_t i = 0; i < count; ++i, p += 4) {
            put_f32(p, distances[first + i]);
        }

        packets.push_back(std::move(out));
    }

    return packets;
}

// UDP link to the Python side on localhost: listens on MY_PORT, replies to PEER_PORT
struct UdpConnection {
    int sock = -1;
    sockaddr_in peer{};
    std::vector<uint8_t> buf = std::vector<uint8_t>(65536);

    bool open();
    void close();
    bool receive(Request& req);
    void send(const std::vector<uint8_t>& out);
    void send(const std::vector<std::vector<uint8_t>>& packets);
};

// creates the socket and binds it to MY_PORT, false on error
inline bool UdpConnection::open() {
    sock = socket(AF_INET, SOCK_DGRAM, 0);

    if (sock < 0) {
        perror("socket");
        return false;
    }

    sockaddr_in me{};
    me.sin_family = AF_INET;
    me.sin_port   = htons(MY_PORT);
    inet_pton(AF_INET, "127.0.0.1", &me.sin_addr);

    if (bind(sock, reinterpret_cast<sockaddr*>(&me), sizeof(me)) < 0) {
        perror("bind");
        close();
        return false;
    }

    peer.sin_family = AF_INET;
    peer.sin_port   = htons(PEER_PORT);
    inet_pton(AF_INET, "127.0.0.1", &peer.sin_addr);

    return true;
}

inline void UdpConnection::close() {
    if (sock >= 0) {
        ::close(sock);
    }
    sock = -1;
}

// blocks until a valid request arrives, malformed packets are ignored
// returns false once Python sends STOP, which is echoed back first
inline bool UdpConnection::receive(Request& req) {
    while (true) {
        ssize_t n = recvfrom(sock, buf.data(), buf.size(), 0, nullptr, nullptr);
        if (n < 0) continue;

        if (n == sizeof(STOP_MSG) && std::memcmp(buf.data(), STOP_MSG, sizeof(STOP_MSG)) == 0) {
            sendto(sock, STOP_MSG, sizeof(STOP_MSG), 0, reinterpret_cast<sockaddr*>(&peer), sizeof(peer));
            return false;
        }

        if (parse_request(buf.data(), static_cast<size_t>(n), req)) return true;
    }
}

inline void UdpConnection::send(const std::vector<uint8_t>& out) {
    sendto(sock, out.data(), out.size(), 0, reinterpret_cast<sockaddr*>(&peer), sizeof(peer));
}

// in order: the EDT window chunks first, the reply last, so Python has the window once the reply arrives
inline void UdpConnection::send(const std::vector<std::vector<uint8_t>>& packets) {
    for (const std::vector<uint8_t>& out : packets) {
        send(out);
    }
}
