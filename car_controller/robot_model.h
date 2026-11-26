#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include "types.h"
#include "params.h"

class RobotModel
{
    public:
        __device__ RobotState kinematicStep(const RobotState &currentState, const ControlInput &u, const ControllerParams &params) const
        {
            RobotState nextState = currentState;
            const double eps = 1e-6;
            const double dt = params.dt;

            double a = (u.speed - currentState.v) / dt;
            if (u.acceleration > 0.0) {
                a = fmax(-u.acceleration, fmin(a, u.acceleration));
            }
            nextState.v = currentState.v + (a * dt);

            // signed distance along the path, exact for constant a
            const double s = 0.5 * (currentState.v + nextState.v) * dt;

            const double beta = atan(params.lr / (params.lf + params.lr) * tan(u.steeringAngle));
            const double kappa = sin(beta) / params.lr;
            const double dPsi = kappa * s;

            // direction of the velocity, psi + beta
            const double course = currentState.theta + beta;

            if (fabs(kappa) < eps) {
                nextState.x += s * cos(course);
                nextState.y += s * sin(course);
            }
            else {
                nextState.x += (sin(course + dPsi) - sin(course)) / kappa;
                nextState.y -= (cos(course + dPsi) - cos(course)) / kappa;
            }
            nextState.theta += dPsi;

            return nextState;
        }

};
