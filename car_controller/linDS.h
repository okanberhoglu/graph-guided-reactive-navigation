#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include "types.h"
#include "params.h"

class LinearDS
{
    public:
        RobotState xGoal;

        __host__ __device__ LinearDS() {}
        __host__ __device__ LinearDS(const RobotState &goal) : xGoal(goal) {}

        __host__ __device__ RobotState getVelocity(const RobotState &x, const ControllerParams &params) const
        {
            RobotState yLin;
            yLin.x = -(x.x - xGoal.x);
            yLin.y = -(x.y - xGoal.y);

            const double dst = sqrt(yLin.x * yLin.x + yLin.y * yLin.y);

            // unit direction outside the ball, dst / attractorRadius inside, so the speed is continuous at the edge
            const double scale = params.nominalSpeed / (dst > params.attractorRadius ? dst : params.attractorRadius);
            yLin.x *= scale;
            yLin.y *= scale;

            return yLin;
        }

};
