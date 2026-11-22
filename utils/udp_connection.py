import math
import socket
import struct
import time
import numpy as np

class UDPConnection:

    def __init__(self, host="127.0.0.1", port=5002, peer_port=5001, reply_timeout=1.0, poll_timeout=0.05, stop_retries=3):
        self.address = (host, port)
        self.peer = (host, peer_port)
        self.reply_timeout = reply_timeout
        self.poll_timeout = poll_timeout
        self.stop_retries = stop_retries

        self.sock = None
        self.seq = 0

        # all C++ controllers compute the EDT themselves (like the ROS controllers from the local map): Python sends the
        # free mask of the sensor window and only draws the EDT window the controller sends back (assemble_edt)
        # request header: seq, x, y, theta, sensor_radius, goal_x, goal_y, rows, cols, is_final (uint8)
        # is_final: 1 when temp_goal is the last one, mppi_controller then stops on it instead of driving through
        # followed by the free mask (rows * cols uint8, 1 = free) and the route nodes (a count + node per node)
        self.request_header = struct.Struct("!I6d2IB")
        # route node (RectRegion of reactive_mppi_controller/types.h): centre x, y, width axis x, y (cos, sin theta), half width, half height
        self.node = struct.Struct("!6d")
        # reply: seq, v, w
        self.reply = struct.Struct("!I2d")
        # then for display: field points (x, y, vx, vy) and kernels (centre x, y, mean x, y), each a count + big-endian float32
        self.count = struct.Struct("!I")

        # car controller (car_reactive_controller/udp_connection.h), same ports, flow and display data:
        # request header: seq, x, y, theta, v, sensor_radius, goal_x, goal_y, rows, cols
        # followed by the free mask (rows * cols uint8, 1 = free) and the route nodes
        self.car_request_header = struct.Struct("!I7d2I")
        # reply: seq, steering angle, target speed, acceleration limit, goal reached (uint8)
        self.car_reply = struct.Struct("!I3dB")
        # kernelGamma of car_reactive_controller/params.h
        self.car_kernel_gamma = 0.002
        # the controller's EDT window (computed in its edt.h, packed in its udp_connection.h; EdtWindow message of the
        # ROS controllers), sent in chunks before the reply by every C++ controller: magic, seq, chunk index,
        # chunk count, first cell, cell count, origin x, y, size x, size y, max distance, robot x, y, theta,
        # robot radius, footprint front, rear, half width, footprint clearance, closest x, y,
        # then cell count big-endian float32 distances
        self.edt_magic = b"EDTW"
        self.edt_header = struct.Struct("!4sIHHII2d2I11d")
        self.edt_chunks = {}  # chunk index -> (header fields, distances) of the current seq
        # a tick brings ~180 KB (EDT + field) at once, give the socket room for it
        self.recv_buffer_bytes = 1 << 22
        self.display_dtype = np.dtype(">f4")
        self.display_columns = 4

        self.recv_size = 65536
        # largest payload a single IPv4 UDP datagram can carry
        self.max_datagram = 65507

        # kernelGamma of reactive_mppi_controller/params.h, only used to shade the area each kernel acts on
        self.kernel_gamma = 0.06
        # sent once the final goal is reached, the controller echoes it back and shuts down
        self.stop_msg = b"STOP"

    def connect(self):
        # UDP has no real connection: bind our port so the controller can reply to it
        if self.sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(self.address)
        except OSError:
            sock.close()
            raise

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buffer_bytes)
        sock.settimeout(self.poll_timeout)
        self.sock = sock
        self.seq = 0

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def parse_display(self, data, offset=None):
        # field and kernels after the command (which ends at offset), empty when the controller sends only the command; None if malformed
        offset = self.reply.size if offset is None else offset
        if len(data) == offset:
            empty = np.zeros((0, self.display_columns))
            return empty, empty.copy()

        row_size = self.display_columns * self.display_dtype.itemsize
        blocks = []

        for _ in range(2):
            if len(data) < offset + self.count.size:
                return None

            (count,) = self.count.unpack_from(data, offset)
            offset += self.count.size

            if len(data) < offset + row_size * count:
                return None

            block = np.frombuffer(data, dtype=self.display_dtype, count=self.display_columns * count, offset=offset)
            blocks.append(block.reshape(count, self.display_columns).astype(float))
            offset += row_size * count

        if offset != len(data):
            return None

        field, kernels = blocks

        return field, kernels

    def pack_nodes(self, nodes):
        nodes = [] if nodes is None else list(nodes)
        parts = [self.count.pack(len(nodes))]

        for node in nodes:
            parts.append(self.node.pack(float(node.center_x), float(node.center_y),
                                        math.cos(node.theta), math.sin(node.theta),
                                        float(node.half_width), float(node.half_height)))

        return b"".join(parts)

    def pack_request(self, robot_pose, temp_goal, sensor_radius, free_mask, nodes, is_final=False):
        # free_mask: the sensor window indexed [x, y], True = known free
        free_mask = np.asarray(free_mask)
        if free_mask.ndim != 2:
            raise ValueError(f"free_mask must be 2D, got shape {free_mask.shape}")

        rows, cols = free_mask.shape

        # header (with the current temp goal) + free mask as uint8 + route nodes
        header = self.request_header.pack(self.seq,
                                          float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]),
                                          float(sensor_radius),
                                          float(temp_goal[0]), float(temp_goal[1]),
                                          rows, cols, int(bool(is_final)))
        packet = header + free_mask.astype(np.uint8).tobytes() + self.pack_nodes(nodes)

        if len(packet) > self.max_datagram:
            raise ValueError(f"request is {len(packet)} bytes, more than one UDP datagram can carry ({self.max_datagram}); use a smaller sensor radius")

        return packet

    def parse_reply(self, data):
        # (u, field, kernels, edt) for the reply to the current request, None for stale or malformed packets
        # edt: the controller's EDT window (assemble_edt), None if one of its chunks was lost
        if len(data) < self.reply.size:
            return None

        received_seq, v, w = self.reply.unpack_from(data)
        if received_seq != self.seq:
            return None

        display = self.parse_display(data)
        if display is None:
            return None

        field, kernels = display

        return [v, w], field, kernels, self.assemble_edt()

    def pack_car_request(self, robot_pose, temp_goal, sensor_radius, free_mask, nodes):
        # robot_pose = (x, y, theta, v); free_mask: the sensor window indexed [x, y], True = known free
        free_mask = np.asarray(free_mask)
        if free_mask.ndim != 2:
            raise ValueError(f"free_mask must be 2D, got shape {free_mask.shape}")
        if len(robot_pose) < 4:
            raise ValueError("the car controller needs robot_pose = (x, y, theta, v)")

        rows, cols = free_mask.shape

        header = self.car_request_header.pack(self.seq,
                                              float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]),
                                              float(robot_pose[3]),
                                              float(sensor_radius),
                                              float(temp_goal[0]), float(temp_goal[1]),
                                              rows, cols)
        packet = header + free_mask.astype(np.uint8).tobytes() + self.pack_nodes(nodes)

        if len(packet) > self.max_datagram:
            raise ValueError(f"request is {len(packet)} bytes, more than one UDP datagram can carry ({self.max_datagram}); use a smaller sensor radius")

        return packet

    def store_edt_chunk(self, data):
        # keeps a chunk of the controller's EDT window if it belongs to the current request
        if len(data) < self.edt_header.size:
            return

        fields = self.edt_header.unpack_from(data)
        _, seq, index, total, first, count = fields[:6]
        if seq != self.seq or index >= total or len(data) != self.edt_header.size + 4 * count:
            return

        distances = np.frombuffer(data, dtype=self.display_dtype, count=count, offset=self.edt_header.size)
        self.edt_chunks[index] = (fields, distances)

    def assemble_edt(self):
        # the controller's EDT window of the current request as a dict, None when a chunk is missing
        if not self.edt_chunks:
            return None

        fields = next(iter(self.edt_chunks.values()))[0]
        total = fields[3]
        origin_x, origin_y, size_x, size_y = fields[6:10]
        (max_distance, robot_x, robot_y, robot_theta, robot_radius,
         footprint_front, footprint_rear, footprint_half_width,
         footprint_clearance, closest_x, closest_y) = fields[10:21]

        if len(self.edt_chunks) != total:
            return None

        distances = np.empty(size_x * size_y, dtype=float)
        filled = 0
        for index in range(total):
            chunk_fields, chunk = self.edt_chunks[index]
            first, count = chunk_fields[4], chunk_fields[5]
            if first + count > distances.size:
                return None
            distances[first:first + count] = chunk
            filled += count

        if filled != distances.size:
            return None

        return {
            "distances": distances.reshape(size_x, size_y),  # [x, y], px, 0 = blocked
            "origin": (origin_x, origin_y),  # centre of cell [0, 0]
            "max_distance": max_distance,
            "robot_pose": (robot_x, robot_y, robot_theta),
            "robot_radius": robot_radius,  # circular robot, 0 for the car
            # car rectangle (front, rear, half width) along the heading, None for a circular robot
            "footprint": (footprint_front, footprint_rear, footprint_half_width) if footprint_half_width > 0.0 else None,
            "footprint_clearance": footprint_clearance,
            "closest": (closest_x, closest_y),
        }

    def parse_car_reply(self, data):
        # (u, field, kernels, goal_reached, edt) for the reply to the current request, None for stale or malformed packets
        # u = [steering angle, target speed, acceleration limit], the input of ackerman_dd_step
        # edt: the controller's EDT window (assemble_edt), None if one of its chunks was lost
        if len(data) < self.car_reply.size:
            return None

        received_seq, steering, speed, acceleration, goal_reached = self.car_reply.unpack_from(data)
        if received_seq != self.seq:
            return None

        display = self.parse_display(data, offset=self.car_reply.size)
        if display is None:
            return None

        field, kernels = display

        return [steering, speed, acceleration], field, kernels, bool(goal_reached), self.assemble_edt()

    def controller_messaging(self, robot_pose, temp_goal, sensor_radius, free_mask, nodes, robot_viz=None, is_final=False):
        # dd controllers (mppi_controller, reactive_controller): robot_pose = (x, y, theta)
        # free_mask: the sensor window indexed [x, y], True = known free; the controller computes the EDT from it
        # nodes: the robot's current route node first, then the next ones up to the temp goal (can be empty)
        # is_final: temp_goal is the last one, the robot has to stop on it (mppi_controller slows down and stops)
        # returns (u, field, kernels, edt), u = [v, w], edt: the EDT window the controller used (assemble_edt)
        # blocks until the controller answers this request, resending it every reply_timeout seconds
        self.seq = (self.seq + 1) & 0xFFFFFFFF  # seq is sent as uint32
        self.edt_chunks = {}
        packet = self.pack_request(robot_pose, temp_goal, sensor_radius, free_mask, nodes, is_final)

        return self._exchange(packet, self.parse_reply, robot_viz)

    def car_controller_messaging(self, robot_pose, temp_goal, sensor_radius, free_mask, nodes, robot_viz=None):
        # same as controller_messaging for the car controller: robot_pose = (x, y, theta, v)
        # returns (u, field, kernels, goal_reached, edt), u = [steering angle, target speed, acceleration limit]
        # goal_reached: the front axle is at the temp goal and the car is braking, until a different goal is sent
        # edt: the EDT window the controller computed and used on this tick (assemble_edt), for display
        self.seq = (self.seq + 1) & 0xFFFFFFFF  # seq is sent as uint32
        self.edt_chunks = {}
        packet = self.pack_car_request(robot_pose, temp_goal, sensor_radius, free_mask, nodes)

        return self._exchange(packet, self.parse_car_reply, robot_viz)

    def _exchange(self, packet, parse, robot_viz=None):
        # sends the request and blocks until parse accepts a reply, resending it every reply_timeout seconds
        # the EDT window chunks the controller sends before its reply are collected on the way
        sock = self._require_socket()

        sock.sendto(packet, self.peer)
        last_sent = time.monotonic()

        while True:
            try:
                data, _ = sock.recvfrom(self.recv_size)
            except socket.timeout:
                data = None
                # keep the window alive while the controller is not answering
                if robot_viz is not None:
                    robot_viz.process_events()
            except ConnectionError:
                # some platforms report "port unreachable" here while the controller is not running yet
                data = None

            if data is not None and data[:len(self.edt_magic)] == self.edt_magic:
                self.store_edt_chunk(data)
            elif data is not None:
                result = parse(data)
                if result is not None:
                    return result

            # checked on every loop, so a stream of stale packets cannot block the resend
            if time.monotonic() - last_sent >= self.reply_timeout:
                sock.sendto(packet, self.peer)
                last_sent = time.monotonic()
                print(f"[Py] no reply yet, resent seq {self.seq}", flush=True)

    def controller_stop(self):
        # returns True once the controller echoes STOP back
        sock = self._require_socket()

        for _ in range(self.stop_retries):
            sock.sendto(self.stop_msg, self.peer)
            deadline = time.monotonic() + self.reply_timeout

            while time.monotonic() < deadline:
                try:
                    data, _ = sock.recvfrom(self.recv_size)
                except (socket.timeout, ConnectionError):
                    continue

                if data == self.stop_msg:
                    print("[Py] controller stopped", flush=True)
                    return True
                # anything else is a late reply to an earlier request, skip it

        print("[Py] no stop acknowledgement from the controller", flush=True)
        return False

    def _require_socket(self):
        if self.sock is None:
            raise RuntimeError("UDPConnection is not connected, call connect() first")
        return self.sock
