#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include "types.h"
#include "params.h"

__device__ inline double goalCost(const Vec2 &pEnd, const RobotState &goal)
{
    const double dx = goal.x - pEnd.x;
    const double dy = goal.y - pEnd.y;

    return sqrt(dx * dx + dy * dy);
}

__device__ inline double collisionCost(const double clearance)
{
    return clearance > 0.0 ? 0.0 : 1.0;
}

__device__ inline double stagnationCost(const double goalDist, const Vec2 &pStart, const Vec2 &pEnd, const ControllerParams &params)
{
    const double dx = pEnd.x - pStart.x;
    const double dy = pEnd.y - pStart.y;

    return goalDist / fmax(params.stayEps, sqrt(dx * dx + dy * dy));
}

__device__ inline double dsCost(const double clearance, const Vec2 &pDot, const Vec2 &fNominal)
{
    const double dx = pDot.x - fNominal.x;
    const double dy = pDot.y - fNominal.y;

    return clearance * sqrt(dx * dx + dy * dy);
}

__device__ inline double totalCost(const double goalDist, const double collisionSteps, const double stagnation, const double dsSum, const ControllerParams &params)
{
    return params.wGoal * goalDist + params.wCollision * collisionSteps + params.wStay * stagnation + params.wDS * dsSum;
}

__device__ inline double nodeDistanceSq(const double x, const double y, const RectRegion &node)
{
    const double dx = x - node.center.x;
    const double dy = y - node.center.y;

    // local coordinates along axis1 (width) and axis2 = (-axis1.y, axis1.x) (height)
    const double u = dx * node.axis1.x + dy * node.axis1.y;
    const double v = -dx * node.axis1.y + dy * node.axis1.x;

    const double outsideU = fmax(fabs(u) - node.halfWidth, 0.0);
    const double outsideV = fmax(fabs(v) - node.halfHeight, 0.0);

    return outsideU * outsideU + outsideV * outsideV;
}

__device__ inline double funnelCost(const RobotState state, const RectRegion *nodes, const int nodeNum, const ControllerParams params)
{
    if (nodeNum <= 0) {
        return 0.0;
    }

    double nearestSq = nodeDistanceSq(state.x, state.y, nodes[0]);

    for (int k = 1; k < nodeNum && nearestSq > 0.0; ++k) {
        nearestSq = fmin(nearestSq, nodeDistanceSq(state.x, state.y, nodes[k]));
    }

    return params.wFunnel * nearestSq;
}