#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include "types.h"

class RobotModel
{
    public:
        __device__ RobotState kinematicStep(const RobotState &currentState, const ControlInput &u, const double dt) const
        {
            RobotState nextState = currentState;
            const double eps = 1e-6;

            const double v = u.v;
            const double w = u.omega;
            const double th = currentState.theta;

            if (fabs(w) < eps) {
                nextState.x += v * dt * cos(th);
                nextState.y += v * dt * sin(th);
            }
            else {
                nextState.theta += w * dt;
                nextState.x += (v / w) * (sin(nextState.theta) - sin(th));
                nextState.y -= (v / w) * (cos(nextState.theta) - cos(th));
            }

            return nextState;
        }
};