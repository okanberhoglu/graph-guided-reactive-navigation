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

// point the field drives, lookAhead px in front of the robot centre
__host__ __device__ inline Vec2 lookAheadPoint(const RobotState &s, const ControllerParams &params)
{
    Vec2 p;
    p.x = s.x + params.lookAhead * cos(s.theta);
    p.y = s.y + params.lookAhead * sin(s.theta);
    return p;
}

// p = c + l h(theta) gives pDot = v h + l omega h_perp, so v = pDot . h and omega = pDot . h_perp / l
// saturation scales v and omega together, which keeps the direction of pDot
__host__ __device__ inline ControlInput lookAheadControl(const RobotState &s, const Vec2 &pDot, const ControllerParams &params)
{
    const double c = cos(s.theta);
    const double sn = sin(s.theta);

    ControlInput u;
    u.v = pDot.x * c + pDot.y * sn;
    u.omega = (-pDot.x * sn + pDot.y * c) / params.lookAhead;

    double scale = 1.0;
    if (u.v > params.vMax) {
        scale = fmin(scale, params.vMax / u.v);
    }
    if (u.v < params.vMin) {
        scale = fmin(scale, params.vMin / u.v);
    }
    if (u.omega > params.omegaMax) {
        scale = fmin(scale, params.omegaMax / u.omega);
    }
    if (u.omega < params.omegaMin) {
        scale = fmin(scale, params.omegaMin / u.omega);
    }
    u.v *= scale;
    u.omega *= scale;

    return u;
}


// eq. 34 at the look-ahead point p: M (f + alpha M_perp g) + f_rep, with the speed normalised to |f| as in the authors' code
// dirs holds one direction per kernel (the sampled ones in a rollout, the means for the command)
// kernelActMax, if not null, keeps the largest rbf_k * alpha seen for every kernel (for the update mask)
__host__ __device__ inline Vec2 fieldVelocity(const Vec2 &p, const RobotState &goal, const EdtMap &edt, const NavKernel *kernels, const Vec2 *dirs, int kernelNum, const ControllerParams &params, double *kernelActMax, FieldInfo &info)
{
    // nominal DS (eq. 36)
    RobotState q;
    q.x = p.x;
    q.y = p.y;
    const RobotState fq = LinearDS(goal).getVelocity(q, params);
    Vec2 f;
    f.x = fq.x;
    f.y = fq.y;
    const double fNorm = vecNorm(f);

    info.nominal = f;
    info.clearance = clearance(edt, p, params);
    info.alignment = 0.0;

    // outside the active zone, or without a defined normal, M = I and the kernels are off
    const Vec2 grad = edtGradient(edt, p);
    const double gradNorm = vecNorm(grad);
    if (info.clearance >= params.dActive || gradNorm < params.normalEps) {
        return f;
    }

    // E = [n t] (eq. 6), n points away from the obstacle
    Vec2 n;
    n.x = grad.x / gradNorm;
    n.y = grad.y / gradNorm;
    Vec2 t;
    t.x = -n.y;
    t.y = n.x;

    info.alignment = fNorm > 0.0 ? vecDot(n, f) / fNorm : 0.0;

    // eq. 13, with lambdaN exactly 0 at and below the safe clearance
    double lambdaN = generalizedSigmoid(info.clearance, 0.0, 1.0, params.dSafe, params.dActive, params.modSteepness);
    double lambdaTau = generalizedSigmoid(info.clearance, params.lambdaTauMax, 1.0, params.dSafe, params.dActive, params.modSteepness);
    if (info.clearance <= params.dSafe) {
        lambdaN = 0.0;
        lambdaTau = params.lambdaTauMax;
    }

    // tail effect (eq. 29-30): no normal damping when the nominal field points away from the obstacle
    const double lambdaV = 1.0 / (1.0 + exp(-params.tailSteepness * info.alignment));
    const double lambdaNTail = lambdaV + (1.0 - lambdaV) * lambdaN;

    // kernel activation (eq. 15, 31), ramps from 0 at the attractor ball to 1 one ball radius further out
    Vec2 toGoal;
    toGoal.x = p.x - goal.x;
    toGoal.y = p.y - goal.y;
    const double alphaGoal = fmin(1.0, fmax(0.0, (vecNorm(toGoal) - params.attractorRadius) / params.attractorRadius));
    const double alpha = (1.0 - lambdaN) * (1.0 - lambdaV) * alphaGoal;

    // navigation kernels (eq. 14, Gaussian RBF as in the authors' code)
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

    // tangential part of g only (eq. 32), scaled by the nominal speed so the directions are relative to it
    const double gTangent = alpha * fNorm * vecDot(t, g);
    Vec2 v;
    v.x = f.x + gTangent * t.x;
    v.y = f.y + gTangent * t.y;

    // M v = lambdaN' (n . v) n + lambdaTau (t . v) t (eq. 5)
    const double vn = lambdaNTail * vecDot(n, v);
    const double vt = lambdaTau * vecDot(t, v);
    Vec2 pDot;
    pDot.x = vn * n.x + vt * t.x;
    pDot.y = vn * n.y + vt * t.y;

    // back to the nominal speed unless the modulation has nearly stopped the motion (authors' code)
    const double speed = vecNorm(pDot);
    if (speed > params.speedNormRatio * fNorm && speed > 0.0) {
        pDot.x *= fNorm / speed;
        pDot.y *= fNorm / speed;
    }

    // repulsion (eq. 33): pushes back out by the margin violation within one step
    if (info.clearance < params.dSafe) {
        const double push = (params.dSafe - info.clearance) / params.dt;
        pDot.x += push * n.x;
        pDot.y += push * n.y;
    }

    return pDot;
}
