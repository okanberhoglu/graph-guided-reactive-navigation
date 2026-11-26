#pragma once

// Lincoln MKZ 2020 in CARLA, kinematic bicycle model
// lengths are in px (map meters / cell_size, 0.1 m), speeds in px/s, angles in rad
struct ControllerParams
{
    double dt = 0.1;

    int horizon = 21;
    int sampleNum = 512;

    // MPPI temperature:
    // w_i = exp(-(J_i - J_min) / beta), beta = mean(J) / betaRatio
    double betaRatio = 50.0;

    double vMin = -15.0;  // -1.5 m/s
    double vMax = 30.0;   //  3.0 m/s

    double maxAccel = 20.0;  // 2 m/s^2, speeding up
    double maxDecel = 40.0;  // 4 m/s^2, slowing down

    double maxSteer = 0.7;  // 40 deg, turning radius 3.4 m at the rear axle

    double reverseAngle = 2.0;       // 115 deg, forward is preferred
    double reverseHysteresis = 0.5;  // 29 deg

    double lf = 14.3;
    double lr = 14.3;

    double carFront = 24.4;         // origin to the front bumper
    double carRear = 24.5;          // origin to the rear bumper
    double carHalfWidth = 9.2;
    double footprintMargin = 2.0;   // added all around: the EDT grid error and a safety distance
    double footprintStep = 1.0;     // spacing of the points on the rectangle's outline where the EDT is read

    double lookAhead = lf;

    double nominalSpeed = 20.0;  // 2.0 m/s
    // ball around the goal: no normalisation, kernels off; the field slows down linearly inside it
    double attractorRadius = 15.0;
    // the car stops once the look-ahead point is this close to the goal
    double goalTolerance = 2.0;
   
    double speedNormRatio = 1.0;

    double dSafe = 3.0;          // lambdaN = 0 at or below this clearance
    double dActive = 20.0;       // modulation is identity above this clearance
    double lambdaTauMax = 2.0;   // tangential gain at the boundary
    double modSteepness = 0.26;  // sigmoid k (1/px), the same shape over dSafe..dActive as before
    double tailSteepness = 100.0;  // tail-effect sigmoid (eq. 30)
    double normalEps = 1e-3;     // below this gradient norm the normal is undefined

    int maxKernels = 16;
    double kernelGamma = 0.002;  // Gaussian kernel width exp(-gamma d^2), 1/px^2, std 1.6 m
    double sigmaKernel = 1.0;    // sampling std of each kernel direction component, relative to the nominal speed
    double alphaMu = 0.5;        // mean learning rate (eq. 18)
    double kernelUpdateThr = 1e-3;  // kernels with a lower mean activation in the rollouts keep their mean (authors' code)

    double placeGoalDist = 30.0;    // farther than this from the goal
    double placeClearance = 20.0;   // closer than this to an obstacle
    double placeNormalAlign = 0.3;  // n . fHat < -1 + this
    double placeKernelDist = 30.0;  // farther than this from existing kernels
    double placeNearDist = 4.0;     // the candidate closest to the look-ahead point is taken when within this, else a random one (authors' code)

    double fieldStep = 12.0;  // px between the displayed field arrows: the robot's 3 px times the car's 4x map scale

    double wGoal = 1.0;
    double wCollision = 1000.0; 
    double wStay = 10.0;
    double wDS = 0.01;           
    double wFunnel = 1.0;    
    double stayEps = 1.0;

};
