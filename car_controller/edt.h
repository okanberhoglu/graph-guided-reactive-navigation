#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include "types.h"
#include "params.h"

struct EdtWindowInfo
{
    double originX = 0.0, originY = 0.0;  // centre of cell [0][0]
    uint32_t sizeX = 0, sizeY = 0;
    double maxDistance = 0.0;  // sensor radius
    double robotX = 0.0, robotY = 0.0, robotTheta = 0.0;  // pose the controller used
    double robotRadius = 0.0;  // circular robot, 0 for the car
    double footprintFront = 0.0, footprintRear = 0.0, footprintHalfWidth = 0.0;  // car rectangle, 0 for a circle
    double footprintClearance = 0.0;  // smallest distance from the footprint to an obstacle
    double closestX = 0.0, closestY = 0.0;  // footprint point where it is
};

inline EdtMap makeEdtMap(const float *data, uint32_t rows, uint32_t cols, const RobotState &pose, double sensorRadius)
{
    EdtMap edt;
    edt.data = data;
    edt.rows = static_cast<int>(rows);
    edt.cols = static_cast<int>(cols);
    edt.cx = static_cast<int>(std::nearbyint(pose.x));
    edt.cy = static_cast<int>(std::nearbyint(pose.y));
    edt.R = static_cast<int>(std::ceil(sensorRadius));
    edt.sensorRadius = sensorRadius;
    return edt;
}

inline void squaredDistance1d(const double *f, int n, double *d, int *v, double *z)
{
    const double inf = std::numeric_limits<double>::infinity();
    int k = 0;
    v[0] = 0;
    z[0] = -inf;
    z[1] = inf;

    for (int q = 1; q < n; ++q) {
        double s;
        while (true) {
            const int p = v[k];
            s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * (q - p));
            if (s > z[k]) {
                break;
            }
            --k;
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = inf;
    }

    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) {
            ++k;
        }
        const int p = v[k];
        d[q] = (q - p) * (q - p) + f[p];
    }
}

inline std::vector<float> distanceTransform(const std::vector<unsigned char> &isSeed, int rows, int cols, double maxDist)
{
    // squared distance of a free cell before the transform, finite so the envelope never computes inf - inf
    constexpr double farSq = 1e20;

    std::vector<double> sq(isSeed.size());
    for (size_t i = 0; i < isSeed.size(); ++i) {
        sq[i] = isSeed[i] ? 0.0 : farSq;
    }

    const int n = rows > cols ? rows : cols;
    std::vector<double> f(n);
    std::vector<double> d(n);
    std::vector<int> v(n);
    std::vector<double> z(n + 1);

    // along b inside every row a, then along a inside every column b
    for (int a = 0; a < rows; ++a) {
        double *row = &sq[static_cast<size_t>(a) * cols];
        squaredDistance1d(row, cols, d.data(), v.data(), z.data());
        for (int b = 0; b < cols; ++b) {
            row[b] = d[b];
        }
    }
    for (int b = 0; b < cols; ++b) {
        for (int a = 0; a < rows; ++a) {
            f[a] = sq[static_cast<size_t>(a) * cols + b];
        }
        squaredDistance1d(f.data(), rows, d.data(), v.data(), z.data());
        for (int a = 0; a < rows; ++a) {
            sq[static_cast<size_t>(a) * cols + b] = d[a];
        }
    }

    std::vector<float> out(sq.size());
    for (size_t i = 0; i < sq.size(); ++i) {
        out[i] = static_cast<float>(std::fmin(std::sqrt(sq[i]), maxDist));
    }
    return out;
}

inline std::vector<float> sensorWindowEdt(const std::vector<unsigned char> &isFree, uint32_t rows, uint32_t cols, const RobotState &pose, double sensorRadius)
{
    const EdtMap window = makeEdtMap(nullptr, rows, cols, pose, sensorRadius);
    const double radiusSq = sensorRadius * sensorRadius;

    std::vector<unsigned char> isSeed(isFree.size(), 0);
    for (int a = 0; a < window.rows; ++a) {
        const double dx = window.cx - window.R + a - pose.x;
        for (int b = 0; b < window.cols; ++b) {
            const double dy = window.cy - window.R + b - pose.y;
            const size_t i = static_cast<size_t>(a) * window.cols + b;
            isSeed[i] = !isFree[i] && dx * dx + dy * dy <= radiusSq;
        }
    }

    return distanceTransform(isSeed, window.rows, window.cols, sensorRadius);
}

__host__ __device__ inline double edtCell(const EdtMap &edt, int a, int b)
{
    if (a < 0 || a >= edt.rows || b < 0 || b >= edt.cols) {
        return edt.sensorRadius;
    }
    return edt.data[a * edt.cols + b];
}

__host__ __device__ inline double edtValue(const EdtMap &edt, const Vec2 &p)
{
    const double fx = p.x - (edt.cx - edt.R);
    const double fy = p.y - (edt.cy - edt.R);
    const int a = static_cast<int>(floor(fx));
    const int b = static_cast<int>(floor(fy));
    const double tx = fx - a;
    const double ty = fy - b;

    return (1.0 - tx) * (1.0 - ty) * edtCell(edt, a, b) + tx * (1.0 - ty) * edtCell(edt, a + 1, b)
         + (1.0 - tx) * ty * edtCell(edt, a, b + 1) + tx * ty * edtCell(edt, a + 1, b + 1);
}

__host__ __device__ inline Vec2 edtGradient(const EdtMap &edt, const Vec2 &p)
{
    Vec2 xPlus = p;
    Vec2 xMinus = p;
    Vec2 yPlus = p;
    Vec2 yMinus = p;
    xPlus.x += 1.0;
    xMinus.x -= 1.0;
    yPlus.y += 1.0;
    yMinus.y -= 1.0;

    Vec2 grad;
    grad.x = 0.5 * (edtValue(edt, xPlus) - edtValue(edt, xMinus));
    grad.y = 0.5 * (edtValue(edt, yPlus) - edtValue(edt, yMinus));
    return grad;
}

struct FootprintOutline
{
    double cornerX[4];
    double cornerY[4];
    int edgePoints[4];
    int count;

    __host__ __device__ explicit FootprintOutline(const ControllerParams &params)
    {
        const double front = params.carFront + params.footprintMargin;
        const double rear = -(params.carRear + params.footprintMargin);
        const double half = params.carHalfWidth + params.footprintMargin;
        cornerX[0] = front;
        cornerY[0] = half;
        cornerX[1] = rear;
        cornerY[1] = half;
        cornerX[2] = rear;
        cornerY[2] = -half;
        cornerX[3] = front;
        cornerY[3] = -half;

        count = 0;
        for (int e = 0; e < 4; ++e) {
            const double dx = cornerX[(e + 1) % 4] - cornerX[e];
            const double dy = cornerY[(e + 1) % 4] - cornerY[e];
            const int n = static_cast<int>(ceil(sqrt(dx * dx + dy * dy) / params.footprintStep));
            edgePoints[e] = n > 1 ? n : 1;
            count += edgePoints[e];
        }
    }

    __host__ __device__ Vec2 point(int i) const
    {
        int e = 0;
        while (e < 3 && i >= edgePoints[e]) {
            i -= edgePoints[e];
            ++e;
        }
        const double t = static_cast<double>(i) / edgePoints[e];

        Vec2 p;
        p.x = cornerX[e] + t * (cornerX[(e + 1) % 4] - cornerX[e]);
        p.y = cornerY[e] + t * (cornerY[(e + 1) % 4] - cornerY[e]);
        return p;
    }
};

__host__ __device__ inline Vec2 carOffset(const RobotState &s, const Vec2 &local)
{
    const double c = cos(s.theta);
    const double sn = sin(s.theta);

    Vec2 r;
    r.x = c * local.x - sn * local.y;
    r.y = sn * local.x + c * local.y;
    return r;
}

__host__ __device__ inline double footprintClearance(const EdtMap &edt, const RobotState &s, const ControllerParams &params, Vec2 *grad, Vec2 *closest = nullptr)
{
    const FootprintOutline outline(params);

    double best = 1e300;
    Vec2 bestPoint;

    for (int i = 0; i < outline.count; ++i) {
        const Vec2 r = carOffset(s, outline.point(i));

        Vec2 b;
        b.x = s.x + r.x;
        b.y = s.y + r.y;

        const double d = edtValue(edt, b);
        if (d < best) {
            best = d;
            bestPoint = b;
        }
    }

    if (grad != nullptr) {
        *grad = edtGradient(edt, bestPoint);

        if (best <= 0.0 || grad->x * grad->x + grad->y * grad->y < 1e-6) {
            const double dx = s.x - bestPoint.x;
            const double dy = s.y - bestPoint.y;
            const double norm = sqrt(dx * dx + dy * dy);
            grad->x = dx / norm;
            grad->y = dy / norm;
        }
    }
    if (closest != nullptr) {
        *closest = bestPoint;
    }
    return best;
}

inline EdtWindowInfo edtWindowInfo(const EdtMap &window, const RobotState &pose, const ControllerParams &params)
{
    EdtWindowInfo info;
    info.originX = window.cx - window.R;
    info.originY = window.cy - window.R;
    info.sizeX = static_cast<uint32_t>(window.rows);
    info.sizeY = static_cast<uint32_t>(window.cols);
    info.maxDistance = window.sensorRadius;
    info.robotX = pose.x;
    info.robotY = pose.y;
    info.robotTheta = pose.theta;
    info.footprintFront = params.carFront + params.footprintMargin;
    info.footprintRear = params.carRear + params.footprintMargin;
    info.footprintHalfWidth = params.carHalfWidth + params.footprintMargin;

    Vec2 closest;
    info.footprintClearance = footprintClearance(window, pose, params, nullptr, &closest);
    info.closestX = closest.x;
    info.closestY = closest.y;
    return info;
}
