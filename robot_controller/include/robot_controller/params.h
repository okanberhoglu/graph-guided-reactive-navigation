#pragma once

struct ControllerParams
{
    double dt = 0.1;

    int horizon = 15;
    int sampleNum = 512;

    double betaRatio = 50.0;

    double goalTolerance = 2.0;

    double vMin = -2.0;
    double vMax = 8.0;
    double omegaMin = -4.0;
    double omegaMax = 4.0;

    double robotRadius = 2.0;

    double lookAhead = 1.5;

    double nominalSpeed = 8.0;
    double attractorRadius = 2.0;
    double speedNormRatio = 0.5;

    double dSafe = 0.5;
    double dActive = 4.0;
    double lambdaTauMax = 2.0;
    double modSteepness = 3.0;
    double tailSteepness = 100.0;
    double normalEps = 1e-3;

    int maxKernels = 16;
    double kernelGamma = 0.06;
    double sigmaKernel = 1.0;
    double alphaMu = 0.5;
    double kernelUpdateThr = 1e-3;

    double placeGoalDist = 4.0;
    double placeClearance = 2.0;
    double placeNormalAlign = 0.3;
    double placeKernelDist = 4.0;
    double placeNearDist = 0.5;

    double fieldStep = 3.0;

    double wGoal = 1.0;
    double wCollision = 100.0;
    double wStay = 10.0;
    double wDS = 0.1;
    double wFunnel = 10.0;
    double stayEps = 0.1;
};