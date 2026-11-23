#include "controller.h"
#include "cost.h"
#include "modulation.h"
#include "robot_model.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cstddef>
#include <cstdio>

namespace {

    #define CUDA_CHECK(call)                                                       \
        do {                                                                       \
            const cudaError_t err__ = (call);                                      \
            if (err__ != cudaSuccess) {                                            \
                std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,        \
                             __FILE__, __LINE__, cudaGetErrorString(err__));       \
            }                                                                      \
        } while (0)
    
    __global__ void initRngKernel(curandState *states, int n, unsigned long long seed)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            curand_init(seed, i, 0, &states[i]);
        }
    }

    // samples[i * kernelNum + k] = g_ik ~ N(mu_k, sigma^2 I), sample 0 gets the means unperturbed
    __global__ void sampleKernelsKernel(Vec2 *samples, curandState *states, const NavKernel *kernels, int kernelNum, int sampleNum, double sigma)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= sampleNum) {
            return;
        }
        if (i == 0) {
            for (int k = 0; k < kernelNum; ++k) {
                samples[k] = kernels[k].mu;
            }
            return;
        }
        curandState st = states[i];
        for (int k = 0; k < kernelNum; ++k) {
            const double2 e = curand_normal2_double(&st);
            Vec2 g;
            g.x = kernels[k].mu.x + sigma * e.x;
            g.y = kernels[k].mu.y + sigma * e.y;
            samples[i * kernelNum + k] = g;
        }
        states[i] = st;
    }

    __global__ void rolloutKernel(RobotState x0, RobotState goal, EdtMap edt, const NavKernel *kernels, int kernelNum, const Vec2 *samples, const RectRegion *nodes, int nodeNum, double *costs, Vec2 *candidates, unsigned char *isCandidate, double *kernelActivation, int sampleNum, ControllerParams params)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= sampleNum) {
            return;
        }

        const Vec2 *dirs = kernelNum > 0 ? samples + i * kernelNum : nullptr;
        double *actMax = kernelNum > 0 ? kernelActivation + i * kernelNum : nullptr;
        for (int k = 0; k < kernelNum; ++k) {
            actMax[k] = 0.0;
        }

        const RobotModel model;
        RobotState x = x0;
        const Vec2 pStart = lookAheadPoint(x0, params);
        double collisionSteps = 0.0;
        double dsSum = 0.0;
        double funnelSum = 0.0;

        for (int t = 0; t < params.horizon; ++t) {
            const Vec2 p = lookAheadPoint(x, params);
            FieldInfo info;
            const Vec2 pDot = fieldVelocity(p, goal, edt, kernels, dirs, kernelNum, params, actMax, info);

            collisionSteps += collisionCost(info.clearance);
            dsSum += dsCost(fmax(info.clearance, 0.0), pDot, info.nominal);

            const double gx = p.x - goal.x;
            const double gy = p.y - goal.y;
            bool candidate = gx * gx + gy * gy > params.placeGoalDist * params.placeGoalDist
                          && info.clearance < params.placeClearance
                          && info.alignment < -1.0 + params.placeNormalAlign;
            for (int k = 0; candidate && k < kernelNum; ++k) {
                const double kx = p.x - kernels[k].center.x;
                const double ky = p.y - kernels[k].center.y;
                candidate = kx * kx + ky * ky > params.placeKernelDist * params.placeKernelDist;
            }
            const int idx = i * params.horizon + t;
            candidates[idx] = p;
            isCandidate[idx] = candidate ? 1 : 0;

            x = model.kinematicStep(x, lookAheadControl(x, pDot, params), params.dt);
            funnelSum += funnelCost(x, nodes, nodeNum, params);
        }

        const Vec2 pEnd = lookAheadPoint(x, params);
        collisionSteps += collisionCost(clearance(edt, pEnd, params));
        const double goalDist = goalCost(pEnd, goal);
        costs[i] = totalCost(goalDist, collisionSteps, stagnationCost(goalDist, pStart, pEnd, params), dsSum, params) + funnelSum;
    }

    __global__ void reduceMinKernel(const double *in, int n, double *out)
    {
        extern __shared__ double s[];
        const int tid = threadIdx.x;
        double m = DBL_MAX;
        for (int i = tid; i < n; i += blockDim.x) {
            m = fmin(m, in[i]);
        }
        s[tid] = m;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s[tid] = fmin(s[tid], s[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[0] = s[0];
        }
    }

    __global__ void reduceSumKernel(const double *in, int n, double *out)
    {
        extern __shared__ double s[];
        const int tid = threadIdx.x;
        double acc = 0.0;
        for (int i = tid; i < n; i += blockDim.x) {
            acc += in[i];
        }
        s[tid] = acc;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s[tid] += s[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[0] = s[0];
        }
    }

    __global__ void weightsKernel(const double *costs, double *weights, int n, double betaRatio, const double *reduce)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            const double beta = fmax(reduce[2] / n / betaRatio, 1e-12);
            weights[i] = exp(-(costs[i] - reduce[0]) / beta);
        }
    }

    __global__ void normalizeKernel(double *weights, int n, const double *reduce)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            const double sum = reduce[1];
            weights[i] *= sum > 0.0 ? 1.0 / sum : 0.0;
        }
    }

    __global__ void updateKernelMeansKernel(NavKernel *kernels, const Vec2 *samples, const double *weights, const double *kernelActivation, int kernelNum, int sampleNum, double alphaMu, double updateThr)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= kernelNum) {
            return;
        }

        double meanActivation = 0.0;
        Vec2 weighted;
        for (int i = 0; i < sampleNum; ++i) {
            const double w = weights[i];
            const Vec2 g = samples[i * kernelNum + k];
            weighted.x += w * g.x;
            weighted.y += w * g.y;
            meanActivation += kernelActivation[i * kernelNum + k];
        }
        meanActivation /= sampleNum;

        if (meanActivation <= updateThr || kernelActivation[k] <= updateThr) {
            return;
        }

        NavKernel ker = kernels[k];
        ker.mu.x = (1.0 - alphaMu) * ker.mu.x + alphaMu * weighted.x;
        ker.mu.y = (1.0 - alphaMu) * ker.mu.y + alphaMu * weighted.y;
        kernels[k] = ker;
    }

    inline int gridFor(int work, int blockSize)
    {
        return (work + blockSize - 1) / blockSize;
    }
}

namespace Controller  {

    std::size_t curandStateSize()
    {
        return sizeof(curandState);
    }

    void initRng(void *states, int sampleNum, unsigned long long seed, int blockSize)
    {
        initRngKernel<<<gridFor(sampleNum, blockSize), blockSize>>>(static_cast<curandState *>(states), sampleNum, seed);
        CUDA_CHECK(cudaGetLastError());
    }

    void sampleKernels(Vec2 *samples, void *states, const NavKernel *kernels, int kernelNum, int sampleNum, double sigma, int blockSize)
    {
        if (kernelNum <= 0 || sampleNum <= 0) {
            return;
        }
        sampleKernelsKernel<<<gridFor(sampleNum, blockSize), blockSize>>>(samples, static_cast<curandState *>(states), kernels, kernelNum, sampleNum, sigma);
        CUDA_CHECK(cudaGetLastError());
    }

    void rollout(RobotState x0, RobotState goal, EdtMap edt, const NavKernel *kernels, int kernelNum, const Vec2 *samples, const RectRegion *nodes, int nodeNum, double *costs, Vec2 *candidates, unsigned char *isCandidate, double *kernelActivation, int sampleNum, ControllerParams params, int blockSize)
    {
        if (sampleNum <= 0) {
            return;
        }
        rolloutKernel<<<gridFor(sampleNum, blockSize), blockSize>>>(x0, goal, edt, kernels, kernelNum, samples, nodes, nodeNum, costs, candidates, isCandidate, kernelActivation, sampleNum, params);
        CUDA_CHECK(cudaGetLastError());
    }

    void computeWeights(const double *costs, double *weights, double *reduce, int sampleNum, double betaRatio, int blockSize)
    {
        if (sampleNum <= 0) {
            return;
        }
        const int grid = gridFor(sampleNum, blockSize);
        const size_t shmem = sizeof(double) * blockSize;

        reduceMinKernel<<<1, blockSize, shmem>>>(costs, sampleNum, &reduce[0]);
        reduceSumKernel<<<1, blockSize, shmem>>>(costs, sampleNum, &reduce[2]);
        weightsKernel<<<grid, blockSize>>>(costs, weights, sampleNum, betaRatio, reduce);
        reduceSumKernel<<<1, blockSize, shmem>>>(weights, sampleNum, &reduce[1]);
        normalizeKernel<<<grid, blockSize>>>(weights, sampleNum, reduce);
        CUDA_CHECK(cudaGetLastError());
    }

    void updateKernelMeans(NavKernel *kernels, const Vec2 *samples, const double *weights, const double *kernelActivation, int kernelNum, int sampleNum, double alphaMu, double updateThr, int blockSize)
    {
        if (kernelNum <= 0 || sampleNum <= 0) {
            return;
        }
        updateKernelMeansKernel<<<gridFor(kernelNum, blockSize), blockSize>>>(kernels, samples, weights, kernelActivation, kernelNum, sampleNum, alphaMu, updateThr);
        CUDA_CHECK(cudaGetLastError());
    }

}