#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "controller.h"
#include "modulation.h"
#include "udp_connection.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err__ = (call);                                      \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,        \
                         __FILE__, __LINE__, cudaGetErrorString(err__));       \
        }                                                                      \
    } while (0)

struct MppiController {
    ControllerParams params;
    int blockSize = 256;

    std::vector<NavKernel> kernels;  // navigation kernels, centres in world px
    bool hasGoal = false;  // lastGoal is set
    RobotState lastGoal;  // goal of the previous tick, the kernels are dropped when it changes

    // host copies of the kernel placement candidates
    std::vector<Vec2> candidates;  // sampleNum * horizon
    std::vector<unsigned char> isCandidate;  // sampleNum * horizon

    // device buffers
    float* dEdt = nullptr;  // rows * cols, resized on demand
    size_t edtCapacity = 0;  // cells allocated for dEdt
    RectRegion* dNodes = nullptr;  // funnel route nodes, resized on demand
    size_t nodesCapacity = 0;  // nodes allocated for dNodes
    NavKernel* dKernels = nullptr;  // maxKernels
    Vec2* dSamples = nullptr;  // sampleNum * maxKernels, [i * kernelNum + k]
    double* dKernelActivation = nullptr;  // sampleNum * maxKernels, [i * kernelNum + k]
    Vec2* dCandidates = nullptr;  // sampleNum * horizon, [i * horizon + t]
    unsigned char* dIsCandidate = nullptr;  // sampleNum * horizon, [i * horizon + t]
    double* dCosts = nullptr;  // sampleNum
    double* dWeights = nullptr;  // sampleNum
    double* dReduce = nullptr;  // [0] = min cost, [1] = weight sum, [2] = cost sum
    void* dRandState = nullptr;  // sampleNum curand states

    std::mt19937 rng{std::random_device{}()};  // picks among the kernel placement candidates

    void allocateDevice();
    void freeDevice();
    void reset();
    void uploadInputs(const Request& req);
    int optimizeKernels(const Request& req);
    void addKernel(const Request& req, int rolloutNum);
    ControlInput commandFromMeans(const Request& req) const;
    ControlInput computeControl(const Request& req);
    std::vector<float> fieldSamples(const Request& req) const;
};

void MppiController::allocateDevice() {
    const size_t N = params.sampleNum;
    const size_t T = params.horizon;
    const size_t K = params.maxKernels;

    kernels.clear();
    kernels.reserve(K);
    candidates.assign(N * T, Vec2{});
    isCandidate.assign(N * T, 0);

    CUDA_CHECK(cudaMalloc(&dKernels, sizeof(NavKernel) * K));
    CUDA_CHECK(cudaMalloc(&dSamples, sizeof(Vec2) * N * K));
    CUDA_CHECK(cudaMalloc(&dKernelActivation, sizeof(double) * N * K));
    CUDA_CHECK(cudaMalloc(&dCandidates, sizeof(Vec2) * N * T));
    CUDA_CHECK(cudaMalloc(&dIsCandidate, sizeof(unsigned char) * N * T));
    CUDA_CHECK(cudaMalloc(&dCosts, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&dWeights, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&dReduce, sizeof(double) * 3));
    CUDA_CHECK(cudaMalloc(&dRandState, Controller::curandStateSize() * N));

    const unsigned long long seed = std::random_device{}();
    Controller::initRng(dRandState, params.sampleNum, seed, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MppiController::freeDevice() {
    cudaFree(dEdt);
    cudaFree(dNodes);
    cudaFree(dKernels);
    cudaFree(dSamples);
    cudaFree(dKernelActivation);
    cudaFree(dCandidates);
    cudaFree(dIsCandidate);
    cudaFree(dCosts);
    cudaFree(dWeights);
    cudaFree(dReduce);
    cudaFree(dRandState);

    dEdt = nullptr;
    edtCapacity = 0;
    dNodes = nullptr;
    nodesCapacity = 0;
    dKernels = nullptr;
    dSamples = nullptr;
    dKernelActivation = nullptr;
    dCandidates = nullptr;
    dIsCandidate = nullptr;
    dCosts = nullptr;
    dWeights = nullptr;
    dReduce = nullptr;
    dRandState = nullptr;
}

void MppiController::reset() {
    kernels.clear();
    hasGoal = false;
}

// host -> device: edt and funnel route nodes (buffers grown on demand) and the current kernels
void MppiController::uploadInputs(const Request& req) {
    const size_t cells = req.edt.size();

    if (cells > edtCapacity) {
        cudaFree(dEdt);
        CUDA_CHECK(cudaMalloc(&dEdt, sizeof(float) * cells));
        edtCapacity = cells;
    }

    CUDA_CHECK(cudaMemcpy(dEdt, req.edt.data(), sizeof(float) * cells, cudaMemcpyHostToDevice));

    // funnel route nodes, none when the robot is outside every route node
    const size_t nodeNum = req.nodes.size();

    if (nodeNum > nodesCapacity) {
        cudaFree(dNodes);
        CUDA_CHECK(cudaMalloc(&dNodes, sizeof(RectRegion) * nodeNum));
        nodesCapacity = nodeNum;
    }

    if (nodeNum > 0) {
        CUDA_CHECK(cudaMemcpy(dNodes, req.nodes.data(), sizeof(RectRegion) * nodeNum, cudaMemcpyHostToDevice));
    }

    if (!kernels.empty()) {
        CUDA_CHECK(cudaMemcpy(dKernels, kernels.data(), sizeof(NavKernel) * kernels.size(), cudaMemcpyHostToDevice));
    }
}


int MppiController::optimizeKernels(const Request& req) {
    const int K = static_cast<int>(kernels.size());
    const int N = K > 0 ? params.sampleNum : 1;
    const EdtMap edt = makeEdtMap(dEdt, req.rows, req.cols, req.pose, req.sensor_radius);

    if (K > 0) {
        Controller::sampleKernels(dSamples, dRandState, dKernels, K, N, params.sigmaKernel, blockSize);
    }
    Controller::rollout(req.pose, req.goal, edt, dKernels, K, dSamples, dNodes, static_cast<int>(req.nodes.size()), dCosts, dCandidates, dIsCandidate, dKernelActivation, N, params, blockSize);
    if (K > 0) {
        Controller::computeWeights(dCosts, dWeights, dReduce, N, params.betaRatio, blockSize);
        Controller::updateKernelMeans(dKernels, dSamples, dWeights, dKernelActivation, K, N, params.alphaMu, params.kernelUpdateThr, blockSize);
        CUDA_CHECK(cudaMemcpy(kernels.data(), dKernels, sizeof(NavKernel) * K, cudaMemcpyDeviceToHost));
    }

    const size_t cells = static_cast<size_t>(N) * params.horizon;
    CUDA_CHECK(cudaMemcpy(candidates.data(), dCandidates, sizeof(Vec2) * cells, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(isCandidate.data(), dIsCandidate, sizeof(unsigned char) * cells, cudaMemcpyDeviceToHost));

    return N;
}

void MppiController::addKernel(const Request& req, int rolloutNum) {
    if (static_cast<int>(kernels.size()) >= params.maxKernels) {
        return;
    }

    std::vector<size_t> found;
    const size_t cells = static_cast<size_t>(rolloutNum) * params.horizon;
    for (size_t j = 0; j < cells; ++j) {
        if (isCandidate[j]) {
            found.push_back(j);
        }
    }
    if (found.empty()) {
        return;
    }

    const Vec2 p = lookAheadPoint(req.pose, params);
    size_t closest = found.front();
    double closestDist = -1.0;
    for (size_t j : found) {
        const double dx = candidates[j].x - p.x;
        const double dy = candidates[j].y - p.y;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (closestDist < 0.0 || d < closestDist) {
            closestDist = d;
            closest = j;
        }
    }
    const size_t pick = closestDist < params.placeNearDist ? closest : found[std::uniform_int_distribution<size_t>(0, found.size() - 1)(rng)];

    NavKernel added;
    added.center = candidates[pick];
    double nearestDist = -1.0;
    for (const NavKernel& k : kernels) {
        const double dx = k.center.x - added.center.x;
        const double dy = k.center.y - added.center.y;
        const double d = dx * dx + dy * dy;
        if (nearestDist < 0.0 || d < nearestDist) {
            nearestDist = d;
            added.mu = k.mu;
        }
    }
    kernels.push_back(added);
}


ControlInput MppiController::commandFromMeans(const Request& req) const {
    std::vector<Vec2> means(kernels.size());
    for (size_t k = 0; k < kernels.size(); ++k) {
        means[k] = kernels[k].mu;
    }

    const EdtMap edt = makeEdtMap(req.edt.data(), req.rows, req.cols, req.pose, req.sensor_radius);
    FieldInfo info;
    const Vec2 pDot = fieldVelocity(lookAheadPoint(req.pose, params), req.goal, edt, kernels.data(), means.data(),
                                    static_cast<int>(kernels.size()), params, nullptr, info);

    return lookAheadControl(req.pose, pDot, params);
}

ControlInput MppiController::computeControl(const Request& req) {
    if (hasGoal && (req.goal.x != lastGoal.x || req.goal.y != lastGoal.y)) {
        reset();
    }
    hasGoal = true;
    lastGoal = req.goal;

    uploadInputs(req);  // host -> device: edt, kernels
    const int rolloutNum = optimizeKernels(req);  // device -> host: kernel means, candidates
    addKernel(req, rolloutNum);

    return commandFromMeans(req);
}

std::vector<float> MppiController::fieldSamples(const Request& req) const {
    std::vector<Vec2> means(kernels.size());
    for (size_t k = 0; k < kernels.size(); ++k) {
        means[k] = kernels[k].mu;
    }

    const EdtMap edt = makeEdtMap(req.edt.data(), req.rows, req.cols, req.pose, req.sensor_radius);
    const double step = params.fieldStep;
    const double x0 = std::round(req.pose.x / step) * step;
    const double y0 = std::round(req.pose.y / step) * step;
    const int n = static_cast<int>(std::ceil(req.sensor_radius / step));

    std::vector<float> out;
    for (int i = -n; i <= n; ++i) {
        for (int j = -n; j <= n; ++j) {
            Vec2 p;
            p.x = x0 + i * step;
            p.y = y0 + j * step;

            const double dx = p.x - req.pose.x;
            const double dy = p.y - req.pose.y;
            if (dx * dx + dy * dy > req.sensor_radius * req.sensor_radius || edtValue(edt, p) <= 0.0) {
                continue;
            }

            FieldInfo info;
            const Vec2 v = fieldVelocity(p, req.goal, edt, kernels.data(), means.data(), static_cast<int>(kernels.size()), params, nullptr, info);
            out.push_back(static_cast<float>(p.x));
            out.push_back(static_cast<float>(p.y));
            out.push_back(static_cast<float>(v.x));
            out.push_back(static_cast<float>(v.y));
        }
    }

    return out;
}

int main() {
    MppiController mppi;
    mppi.allocateDevice();

    UdpConnection udp;
    if (!udp.open()) {
        return 1;
    }

    std::printf("[C++] waiting for Python ...\n");
    std::fflush(stdout);

    Request req;

    bool has_last = false;
    uint32_t last_seq = 0;
    std::vector<std::vector<uint8_t>> last_packets;  

    while (udp.receive(req)) {

        if (has_last && req.seq == last_seq) {
            udp.send(last_packets);
            continue;
        }

        if (has_last && req.seq < last_seq) {
            mppi.reset();
        }

        req.edt = sensorWindowEdt(req.isFree, req.rows, req.cols, req.pose, req.sensor_radius);

        const ControlInput u = mppi.computeControl(req);

        has_last = true;
        last_seq = req.seq;
        const EdtMap window = makeEdtMap(req.edt.data(), req.rows, req.cols, req.pose, req.sensor_radius);
        last_packets = build_edt_packets(req.seq, edtWindowInfo(window, req.pose, mppi.params), req.edt);
        last_packets.push_back(build_reply(req.seq, u, mppi.fieldSamples(req), mppi.kernels));

        std::printf("[C++] seq %u pose (%.2f, %.2f, %.2f) goal (%.2f, %.2f) sensor %.1f map %ux%u kernels %zu -> u (%.2f, %.2f)\n",
                    req.seq, req.pose.x, req.pose.y, req.pose.theta, req.goal.x, req.goal.y, req.sensor_radius,
                    req.rows, req.cols, mppi.kernels.size(), u.v, u.omega);
        std::fflush(stdout);

        udp.send(last_packets);
    }

    std::printf("[C++] stop received, shutting down\n");
    std::fflush(stdout);

    udp.close();
    mppi.freeDevice();
}
