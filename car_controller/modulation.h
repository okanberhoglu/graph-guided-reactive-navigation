#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include "types.h"
#include "params.h"
#include "linDS.h"
#include "edt.h"

__host__ __device__ inline double vecDot(const Vec2 &a, const Vec2 &b)
{
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline double vecNorm(const Vec2 &a)
{
    return sqrt(vecDot(a, a));
}

__host__ __device__ inline double generalizedSigmoid(double x, double yLow, double yHigh, double x0, double x1, double k)
{
    return yLow + (yHigh - yLow) / (1.0 + exp(-k * (x - 0.5 * (x0 + x1))));
}

__host__ __device__ inline Vec2 lookAheadPoint(const RobotState &s, const ControllerParams &params)
{
    Vec2 p;
    p.x = s.x + params.lookAhead * cos(s.theta);
    p.y = s.y + params.lookAhead * sin(s.theta);
    return p;
}

__host__ __device__ inline ControlInput lookAheadControl(const RobotState &s, const Vec2 &pDot, const ControllerParams &params)
{
    const double c = cos(s.theta);
    const double sn = sin(s.theta);
    const double along = pDot.x * c + pDot.y * sn;
    const double side = -pDot.x * sn + pDot.y * c;
    const double speed = sqrt(along * along + side * side);

    const double angle = atan2(side, along);

    const double reverseFrom = s.v < 0.0 ? params.reverseAngle - params.reverseHysteresis : params.reverseAngle;
    const bool reverse = fabs(angle) > reverseFrom;

    const double wheel = reverse ? (angle > 0.0 ? angle - M_PI : angle + M_PI) : angle;
    const double delta = fmax(-params.maxSteer, fmin(wheel, params.maxSteer));
    const double vf = speed * fmax(0.0, cos(wheel - delta));
    const double beta = atan(params.lr / (params.lf + params.lr) * tan(delta));

    ControlInput u;
    u.steeringAngle = delta;
    u.speed = (reverse ? -vf : vf) * cos(delta) / cos(beta);
    u.speed = fmax(params.vMin, fmin(u.speed, params.vMax));

    const bool speedingUp = u.speed * s.v >= 0.0 && fabs(u.speed) > fabs(s.v);
    u.acceleration = speedingUp ? params.maxAccel : params.maxDecel;

    return u;
}

__host__ __device__ inline Vec2 fieldVelocity(const RobotState &s, const RobotState &goal, const EdtMap &edt, const NavKernel *kernels, const Vec2 *dirs, int kernelNum, const ControllerParams &params, double *kernelActMax, FieldInfo &info)
{
    const Vec2 p = lookAheadPoint(s, params);

    RobotState q;
    q.x = p.x;
    q.y = p.y;
    const RobotState fq = LinearDS(goal).getVelocity(q, params);
    Vec2 f;
    f.x = fq.x;
    f.y = fq.y;
    const double fNorm = vecNorm(f);

    Vec2 grad;
    info.nominal = f;
    info.clearance = footprintClearance(edt, s, params, &grad);
    info.alignment = 0.0;

    const double gradNorm = vecNorm(grad);
    if (info.clearance >= params.dActive || gradNorm < params.normalEps) {
        return f;
    }

    Vec2 n;
    n.x = grad.x / gradNorm;
    n.y = grad.y / gradNorm;
    Vec2 t;
    t.x = -n.y;
    t.y = n.x;

    info.alignment = fNorm > 0.0 ? vecDot(n, f) / fNorm : 0.0;

    double lambdaN = generalizedSigmoid(info.clearance, 0.0, 1.0, params.dSafe, params.dActive, params.modSteepness);
    double lambdaTau = generalizedSigmoid(info.clearance, params.lambdaTauMax, 1.0, params.dSafe, params.dActive, params.modSteepness);
    if (info.clearance <= params.dSafe) {
        lambdaN = 0.0;
        lambdaTau = params.lambdaTauMax;
    }

    const double lambdaV = 1.0 / (1.0 + exp(-params.tailSteepness * info.alignment));
    const double lambdaNTail = lambdaV + (1.0 - lambdaV) * lambdaN;

    Vec2 toGoal;
    toGoal.x = p.x - goal.x;
    toGoal.y = p.y - goal.y;
    const double alphaGoal = fmin(1.0, fmax(0.0, (vecNorm(toGoal) - params.attractorRadius) / params.attractorRadius));
    const double alpha = (1.0 - lambdaN) * (1.0 - lambdaV) * alphaGoal;

    Vec2 g;
    if (alpha > 0.0) {
        for (int k = 0; k < kernelNum; ++k) {
            const double dx = p.x - kernels[k].center.x;
            const double dy = p.y - kernels[k].center.y;
            const double rbf = exp(-params.kernelGamma * (dx * dx + dy * dy));
            g.x += rbf * dirs[k].x;
            g.y += rbf * dirs[k].y;
            if (kernelActMax != nullptr) {
                kernelActMax[k] = fmax(kernelActMax[k], rbf * alpha);
            }
        }
    }

    const double gTangent = alpha * fNorm * vecDot(t, g);
    Vec2 v;
    v.x = f.x + gTangent * t.x;
    v.y = f.y + gTangent * t.y;

    const double vn = lambdaNTail * vecDot(n, v);
    const double vt = lambdaTau * vecDot(t, v);
    Vec2 pDot;
    pDot.x = vn * n.x + vt * t.x;
    pDot.y = vn * n.y + vt * t.y;

    const double speed = vecNorm(pDot);
    if (speed > params.speedNormRatio * fNorm && speed > 0.0) {
        pDot.x *= fNorm / speed;
        pDot.y *= fNorm / speed;
    }

    if (info.clearance < params.dSafe) {
        const double push = (params.dSafe - info.clearance) / params.dt;
        pDot.x += push * n.x;
        pDot.y += push * n.y;
    }

    return pDot;
}
