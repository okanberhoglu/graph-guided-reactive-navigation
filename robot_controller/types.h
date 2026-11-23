#pragma once

struct RobotState
{
    double x = 0.0; // x position (px)
    double y = 0.0; // y position (px)
    double theta = 0.0; // heading theta (radians)
};

struct ControlInput
{
    double v = 0.0; // linear velocity (px/s)
    double omega = 0.0; // angular velocity (rad/s)
};

struct EdtMap
{
    const float *data = nullptr; // rows x cols, row-major, indexed [x][y]
    int rows = 0;
    int cols = 0;
    int cx = 0; // world cell the window is centred on
    int cy = 0;
    int R = 0;
    double sensorRadius = 0.0;
};

struct Vec2
{
    double x = 0.0;
    double y = 0.0;
};

struct NavKernel
{
    Vec2 center; // q_hat_k, world px
    Vec2 mu; // mean deflection direction, relative to the nominal speed
};

struct RectRegion
{
    Vec2 center; // (center_x, center_y)
    Vec2 axis1; // unit vector along the width, (cos theta, sin theta); axis2 = (-axis1.y, axis1.x)
    double halfWidth  = 0.0; // half extent along axis1
    double halfHeight = 0.0; // half extent along axis2
};

struct FieldInfo
{
    double clearance = 0.0; // Gamma at p
    double alignment = 0.0; // n . fHat, -1 = nominal field points straight into the obstacle
    Vec2 nominal; // f(p)
};
