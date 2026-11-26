#pragma once

#include <cstddef>
#include "types.h"
#include "params.h"

namespace Controller {

    std::size_t curandStateSize();

    void initRng(void *states, int sampleNum, unsigned long long seed, int blockSize);

    // g_ik ~ N(mu_k, sigma^2 I) for every sample i and navigation kernel k, sample 0 gets the means
    void sampleKernels(Vec2 *samples, void *states, const NavKernel *kernels, int kernelNum, int sampleNum, double sigma, int blockSize);

    // integrates the modulated DS for every sample, writes its cost, the kernel placement candidates
    // and the largest activation of every kernel (for the update mask); edt.data and nodes must point to device memory
    void rollout(RobotState x0, RobotState goal, EdtMap edt, const NavKernel *kernels, int kernelNum, const Vec2 *samples, const RectRegion *nodes, int nodeNum, double *costs, Vec2 *candidates, unsigned char *isCandidate, double *kernelActivation, int sampleNum, ControllerParams params, int blockSize);

    // w_i = exp(-(c_i - c_min) / beta) normalised to sum 1, beta = mean cost / betaRatio
    // reduce needs 3 doubles: [0] min cost, [1] weight sum, [2] cost sum
    void computeWeights(const double *costs, double *weights, double *reduce, int sampleNum, double betaRatio, int blockSize);

    // mu_k <- (1 - alphaMu) mu_k + alphaMu sum_i w_i g_ik, only for kernels the rollouts activated above updateThr
    void updateKernelMeans(NavKernel *kernels, const Vec2 *samples, const double *weights, const double *kernelActivation, int kernelNum, int sampleNum, double alphaMu, double updateThr, int blockSize);

}
