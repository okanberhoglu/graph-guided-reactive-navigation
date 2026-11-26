#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <messages/msg/edt_window.hpp>
#include <messages/msg/rect_region_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "controller.h"
#include "edt.h"
#include "modulation.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err__ = (call);                                      \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,        \
                         __FILE__, __LINE__, cudaGetErrorString(err__));       \
        }                                                                      \
    } while (0)

struct ControllerInput {
    RobotState pose;
    RobotState goal;
    double sensor_radius = 0.0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<float> edt;
    std::vector<RectRegion> nodes;
};

struct MppiController {
    ControllerParams params;
    int blockSize = 256;

    std::vector<NavKernel> kernels;
    bool hasGoal = false;
    RobotState lastGoal;

    std::vector<Vec2> candidates;
    std::vector<unsigned char> isCandidate;

    float* dEdt = nullptr;
    size_t edtCapacity = 0;
    RectRegion* dNodes = nullptr;
    size_t nodesCapacity = 0;
    NavKernel* dKernels = nullptr;
    Vec2* dSamples = nullptr;
    double* dKernelActivation = nullptr;
    Vec2* dCandidates = nullptr;
    unsigned char* dIsCandidate = nullptr;
    double* dCosts = nullptr;
    double* dWeights = nullptr;
    double* dReduce = nullptr;
    void* dRandState = nullptr;

    std::mt19937 rng{std::random_device{}()};

    void allocateDevice();
    void freeDevice();
    void reset();
    void uploadInputs(const ControllerInput& req);
    int optimizeKernels(const ControllerInput& req);
    void addKernel(const ControllerInput& req, int rolloutNum);
    ControlInput commandFromMeans(const ControllerInput& req) const;
    ControlInput computeControl(const ControllerInput& req);
};

void MppiController::allocateDevice() {
    const size_t N = params.sampleNum;
    const size_t T = params.horizon;
    const size_t K = params.maxKernels;

    kernels.clear();
    kernels.reserve(K);
    candidates.assign(N * T, Vec2{});
    isCandidate.assign(N * T, 0);

    CUDA_CHECK(cudaMalloc(&dKernels, sizeof(NavKernel) * K));
    CUDA_CHECK(cudaMalloc(&dSamples, sizeof(Vec2) * N * K));
    CUDA_CHECK(cudaMalloc(&dKernelActivation, sizeof(double) * N * K));
    CUDA_CHECK(cudaMalloc(&dCandidates, sizeof(Vec2) * N * T));
    CUDA_CHECK(cudaMalloc(&dIsCandidate, sizeof(unsigned char) * N * T));
    CUDA_CHECK(cudaMalloc(&dCosts, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&dWeights, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&dReduce, sizeof(double) * 3));
    CUDA_CHECK(cudaMalloc(&dRandState, Controller::curandStateSize() * N));

    const unsigned long long seed = std::random_device{}();
    Controller::initRng(dRandState, params.sampleNum, seed, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MppiController::freeDevice() {
    cudaFree(dEdt);
    cudaFree(dNodes);
    cudaFree(dKernels);
    cudaFree(dSamples);
    cudaFree(dKernelActivation);
    cudaFree(dCandidates);
    cudaFree(dIsCandidate);
    cudaFree(dCosts);
    cudaFree(dWeights);
    cudaFree(dReduce);
    cudaFree(dRandState);

    dEdt = nullptr;
    edtCapacity = 0;
    dNodes = nullptr;
    nodesCapacity = 0;
    dKernels = nullptr;
    dSamples = nullptr;
    dKernelActivation = nullptr;
    dCandidates = nullptr;
    dIsCandidate = nullptr;
    dCosts = nullptr;
    dWeights = nullptr;
    dReduce = nullptr;
    dRandState = nullptr;
}

void MppiController::reset() {
    kernels.clear();
    hasGoal = false;
}

void MppiController::uploadInputs(const ControllerInput& req) {
    const size_t cells = req.edt.size();

    if (cells > edtCapacity) {
        cudaFree(dEdt);
        CUDA_CHECK(cudaMalloc(&dEdt, sizeof(float) * cells));
        edtCapacity = cells;
    }

    CUDA_CHECK(cudaMemcpy(dEdt, req.edt.data(), sizeof(float) * cells, cudaMemcpyHostToDevice));

    const size_t nodeNum = req.nodes.size();

    if (nodeNum > nodesCapacity) {
        cudaFree(dNodes);
        CUDA_CHECK(cudaMalloc(&dNodes, sizeof(RectRegion) * nodeNum));
        nodesCapacity = nodeNum;
    }

    if (nodeNum > 0) {
        CUDA_CHECK(cudaMemcpy(dNodes, req.nodes.data(), sizeof(RectRegion) * nodeNum, cudaMemcpyHostToDevice));
    }

    if (!kernels.empty()) {
        CUDA_CHECK(cudaMemcpy(dKernels, kernels.data(), sizeof(NavKernel) * kernels.size(), cudaMemcpyHostToDevice));
    }
}

int MppiController::optimizeKernels(const ControllerInput& req) {
    const int K = static_cast<int>(kernels.size());
    const int N = K > 0 ? params.sampleNum : 1;
    const EdtMap edt = makeEdtMap(dEdt, req.rows, req.cols, req.pose, req.sensor_radius);

    if (K > 0) {
        Controller::sampleKernels(dSamples, dRandState, dKernels, K, N, params.sigmaKernel, blockSize);
    }
    Controller::rollout(req.pose, req.goal, edt, dKernels, K, dSamples, dNodes, static_cast<int>(req.nodes.size()), dCosts, dCandidates, dIsCandidate, dKernelActivation, N, params, blockSize);
    if (K > 0) {
        Controller::computeWeights(dCosts, dWeights, dReduce, N, params.betaRatio, blockSize);
        Controller::updateKernelMeans(dKernels, dSamples, dWeights, dKernelActivation, K, N, params.alphaMu, params.kernelUpdateThr, blockSize);
        CUDA_CHECK(cudaMemcpy(kernels.data(), dKernels, sizeof(NavKernel) * K, cudaMemcpyDeviceToHost));
    }

    const size_t cells = static_cast<size_t>(N) * params.horizon;
    CUDA_CHECK(cudaMemcpy(candidates.data(), dCandidates, sizeof(Vec2) * cells, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(isCandidate.data(), dIsCandidate, sizeof(unsigned char) * cells, cudaMemcpyDeviceToHost));

    return N;
}

void MppiController::addKernel(const ControllerInput& req, int rolloutNum) {
    if (static_cast<int>(kernels.size()) >= params.maxKernels) {
        return;
    }

    std::vector<size_t> found;
    const size_t cells = static_cast<size_t>(rolloutNum) * params.horizon;
    for (size_t j = 0; j < cells; ++j) {
        if (isCandidate[j]) {
            found.push_back(j);
        }
    }
    if (found.empty()) {
        return;
    }

    const Vec2 p = lookAheadPoint(req.pose, params);
    size_t closest = found.front();
    double closestDist = -1.0;
    for (size_t j : found) {
        const double dx = candidates[j].x - p.x;
        const double dy = candidates[j].y - p.y;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (closestDist < 0.0 || d < closestDist) {
            closestDist = d;
            closest = j;
        }
    }
    const size_t pick = closestDist < params.placeNearDist ? closest : found[std::uniform_int_distribution<size_t>(0, found.size() - 1)(rng)];

    NavKernel added;
    added.center = candidates[pick];
    double nearestDist = -1.0;
    for (const NavKernel& k : kernels) {
        const double dx = k.center.x - added.center.x;
        const double dy = k.center.y - added.center.y;
        const double d = dx * dx + dy * dy;
        if (nearestDist < 0.0 || d < nearestDist) {
            nearestDist = d;
            added.mu = k.mu;
        }
    }
    kernels.push_back(added);
}

ControlInput MppiController::commandFromMeans(const ControllerInput& req) const {
    std::vector<Vec2> means(kernels.size());
    for (size_t k = 0; k < kernels.size(); ++k) {
        means[k] = kernels[k].mu;
    }

    const EdtMap edt = makeEdtMap(req.edt.data(), req.rows, req.cols, req.pose, req.sensor_radius);
    FieldInfo info;
    const Vec2 pDot = fieldVelocity(lookAheadPoint(req.pose, params), req.goal, edt, kernels.data(), means.data(),
                                    static_cast<int>(kernels.size()), params, nullptr, info);

    return lookAheadControl(req.pose, pDot, params);
}

ControlInput MppiController::computeControl(const ControllerInput& req) {
    if (hasGoal && (req.goal.x != lastGoal.x || req.goal.y != lastGoal.y)) {
        reset();
    }
    hasGoal = true;
    lastGoal = req.goal;

    uploadInputs(req);
    const int rolloutNum = optimizeKernels(req);
    addKernel(req, rolloutNum);

    return commandFromMeans(req);
}

namespace {

    constexpr int kOccupiedThreshold = 50;

    double yawFromQuaternion(const geometry_msgs::msg::Quaternion& q) {
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }

}

class ReactiveControllerNode : public rclcpp::Node {
public:
    ReactiveControllerNode();
    ~ReactiveControllerNode() override;

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void nodesCallback(const messages::msg::RectRegionArray::SharedPtr msg);
    void localMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void controlStep();

    bool inMapFrame(const std::string& frame, const char* topic);
    bool robotPoseInMap(RobotState& out);
    void buildEdtWindow(ControllerInput& in) const;
    void publishEdt(const ControllerInput& in);

    MppiController m_mppi;

    std::string m_mapFrame;
    double m_cellSize = 0.1;
    double m_sensorRadius = 20.0;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_odomSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr m_goalSub;
    rclcpp::Subscription<messages::msg::RectRegionArray>::SharedPtr m_nodesSub;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr m_localMapSub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr m_cmdPub;
    rclcpp::Publisher<messages::msg::EdtWindow>::SharedPtr m_edtPub;
    rclcpp::TimerBase::SharedPtr m_controlTimer;

    std::shared_ptr<tf2_ros::Buffer> m_tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> m_tfListener;

    RobotState m_odomPose;
    std::string m_odomFrame;
    bool m_haveOdom = false;

    RobotState m_goal;
    bool m_haveGoal = false;

    bool m_goalReached = false;

    std::vector<RectRegion> m_nodes;
    nav_msgs::msg::OccupancyGrid::SharedPtr m_localMap;
};

ReactiveControllerNode::ReactiveControllerNode() : rclcpp::Node("reactive_controller") {
    m_mapFrame = declare_parameter<std::string>("map_frame", "map");
    m_cellSize = declare_parameter<double>("cell_size", 0.1);
    m_sensorRadius = declare_parameter<double>("sensor_radius_cells", 20.0);

    if (m_mppi.params.goalTolerance <= m_mppi.params.lookAhead) {
        RCLCPP_WARN(get_logger(), "goalTolerance %.1f px is not larger than lookAhead %.1f px (params.h): the robot may never get close enough to stop",
                    m_mppi.params.goalTolerance, m_mppi.params.lookAhead);
    }

    if (m_cellSize <= 0.0 || m_sensorRadius <= 0.0) {
        throw std::invalid_argument("cell_size and sensor_radius_cells must be positive");
    }

    m_mppi.allocateDevice();

    m_tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
    m_tfListener = std::make_shared<tf2_ros::TransformListener>(*m_tfBuffer);

    m_cmdPub = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    m_edtPub = create_publisher<messages::msg::EdtWindow>("/controller_edt", 1);

    using std::placeholders::_1;
    m_odomSub = create_subscription<nav_msgs::msg::Odometry>("/odom", 10, std::bind(&ReactiveControllerNode::odomCallback, this, _1));
    m_goalSub = create_subscription<geometry_msgs::msg::PoseStamped>("/temp_goal_point", 10, std::bind(&ReactiveControllerNode::goalCallback, this, _1));
    m_nodesSub = create_subscription<messages::msg::RectRegionArray>("/current_nodes", 10, std::bind(&ReactiveControllerNode::nodesCallback, this, _1));
    m_localMapSub = create_subscription<nav_msgs::msg::OccupancyGrid>("/local_map", 10, std::bind(&ReactiveControllerNode::localMapCallback, this, _1));

    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(m_mppi.params.dt));
    m_controlTimer = create_wall_timer(period, std::bind(&ReactiveControllerNode::controlStep, this));

    RCLCPP_INFO(get_logger(), "reactive_controller started (cell size %.3f m, sensor radius %.1f px): waiting for /odom, /temp_goal_point, /local_map",
                m_cellSize, m_sensorRadius);
}

ReactiveControllerNode::~ReactiveControllerNode() {
    m_mppi.freeDevice();
}

bool ReactiveControllerNode::inMapFrame(const std::string& frame, const char* topic) {
    if (frame.empty() || frame == m_mapFrame) {
        return true;
    }

    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s is in frame '%s', expected '%s'; ignoring it",
                         topic, frame.c_str(), m_mapFrame.c_str());
    return false;
}

void ReactiveControllerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    m_odomPose.x = msg->pose.pose.position.x;
    m_odomPose.y = msg->pose.pose.position.y;
    m_odomPose.theta = yawFromQuaternion(msg->pose.pose.orientation);
    m_odomFrame = msg->header.frame_id.empty() ? "odom" : msg->header.frame_id;
    m_haveOdom = true;
}

void ReactiveControllerNode::goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (!inMapFrame(msg->header.frame_id, "/temp_goal_point")) {
        return;
    }

    const double x = msg->pose.position.x / m_cellSize;
    const double y = msg->pose.position.y / m_cellSize;

    if (!m_haveGoal || x != m_goal.x || y != m_goal.y) {
        RCLCPP_INFO(get_logger(), "temp goal (%.2f, %.2f) m", msg->pose.position.x, msg->pose.position.y);
        m_goalReached = false;
    }

    m_goal.x = x;
    m_goal.y = y;
    m_haveGoal = true;
}

void ReactiveControllerNode::nodesCallback(const messages::msg::RectRegionArray::SharedPtr msg) {
    if (!inMapFrame(msg->header.frame_id, "/current_nodes")) {
        return;
    }

    m_nodes.clear();
    m_nodes.reserve(msg->regions.size());

    for (const auto& region : msg->regions) {
        RectRegion node;
        node.center.x = region.center_x / m_cellSize;
        node.center.y = region.center_y / m_cellSize;
        node.axis1.x = std::cos(region.theta);
        node.axis1.y = std::sin(region.theta);
        node.halfWidth = region.half_width / m_cellSize;
        node.halfHeight = region.half_height / m_cellSize;
        m_nodes.push_back(node);
    }
}

void ReactiveControllerNode::localMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    if (!inMapFrame(msg->header.frame_id, "/local_map")) {
        return;
    }

    m_localMap = msg;
}

bool ReactiveControllerNode::robotPoseInMap(RobotState& out) {
    if (!m_haveOdom) {
        return false;
    }
    if (m_odomFrame == m_mapFrame) {
        out = m_odomPose;
        return true;
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = m_tfBuffer->lookupTransform(m_mapFrame, m_odomFrame, tf2::TimePointZero, tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no TF %s -> %s for the robot pose: %s",
                             m_mapFrame.c_str(), m_odomFrame.c_str(), ex.what());
        return false;
    }

    const double tx = tf.transform.translation.x;
    const double ty = tf.transform.translation.y;
    const double yaw = yawFromQuaternion(tf.transform.rotation);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    out.x = tx + c * m_odomPose.x - s * m_odomPose.y;
    out.y = ty + s * m_odomPose.x + c * m_odomPose.y;
    out.theta = m_odomPose.theta + yaw;
    return true;
}

void ReactiveControllerNode::buildEdtWindow(ControllerInput& in) const {
    const nav_msgs::msg::OccupancyGrid& map = *m_localMap;
    const double resolution = map.info.resolution;
    const double originX = map.info.origin.position.x;
    const double originY = map.info.origin.position.y;
    const int width = static_cast<int>(map.info.width);
    const int height = static_cast<int>(map.info.height);

    const EdtMap window = makeEdtMap(nullptr, 0, 0, in.pose, in.sensor_radius);
    const int size = 2 * window.R + 1;
    const double radiusSq = in.sensor_radius * in.sensor_radius;

    std::vector<unsigned char> isFree(static_cast<size_t>(size) * size, 0);

    for (int a = 0; a < size; ++a) {
        for (int b = 0; b < size; ++b) {
            const double x = window.cx - window.R + a;
            const double y = window.cy - window.R + b;

            const double dx = x - in.pose.x;
            const double dy = y - in.pose.y;
            if (dx * dx + dy * dy > radiusSq) {
                continue;
            }

            const int gx = static_cast<int>(std::floor((x * m_cellSize - originX) / resolution));
            const int gy = static_cast<int>(std::floor((y * m_cellSize - originY) / resolution));
            if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
                continue;
            }

            const int value = map.data[static_cast<size_t>(gy) * width + gx];
            isFree[static_cast<size_t>(a) * size + b] = value >= 0 && value < kOccupiedThreshold;
        }
    }

    in.rows = static_cast<uint32_t>(size);
    in.cols = static_cast<uint32_t>(size);
    in.edt = sensorWindowEdt(isFree, in.rows, in.cols, in.pose, in.sensor_radius);
}

void ReactiveControllerNode::publishEdt(const ControllerInput& in) {
    if (m_edtPub->get_subscription_count() == 0) {
        return;
    }

    const EdtMap window = makeEdtMap(in.edt.data(), in.rows, in.cols, in.pose, in.sensor_radius);
    const EdtWindowInfo info = edtWindowInfo(window, in.pose, m_mppi.params);

    messages::msg::EdtWindow msg;
    msg.header.stamp = now();
    msg.header.frame_id = m_mapFrame;
    msg.resolution = m_cellSize;
    msg.origin_x = info.originX * m_cellSize;
    msg.origin_y = info.originY * m_cellSize;
    msg.size_x = info.sizeX;
    msg.size_y = info.sizeY;
    msg.max_distance = info.maxDistance * m_cellSize;

    msg.distances.resize(in.edt.size());
    for (size_t i = 0; i < in.edt.size(); ++i) {
        msg.distances[i] = static_cast<float>(in.edt[i] * m_cellSize);
    }

    msg.robot_x = info.robotX * m_cellSize;
    msg.robot_y = info.robotY * m_cellSize;
    msg.robot_theta = info.robotTheta;
    msg.robot_radius = info.robotRadius * m_cellSize;
    msg.footprint_front = info.footprintFront * m_cellSize;
    msg.footprint_rear = info.footprintRear * m_cellSize;
    msg.footprint_half_width = info.footprintHalfWidth * m_cellSize;
    msg.footprint_clearance = info.footprintClearance * m_cellSize;
    msg.closest_x = info.closestX * m_cellSize;
    msg.closest_y = info.closestY * m_cellSize;

    m_edtPub->publish(msg);
}

void ReactiveControllerNode::controlStep() {
    RobotState pose;
    if (!m_haveGoal || m_goalReached || !m_localMap || !robotPoseInMap(pose)) {
        return;
    }

    const double goalDist = std::hypot(m_goal.x - pose.x / m_cellSize, m_goal.y - pose.y / m_cellSize);
    if (goalDist <= m_mppi.params.goalTolerance) {
        m_goalReached = true;
        m_cmdPub->publish(geometry_msgs::msg::Twist());
        RCLCPP_INFO(get_logger(), "goal (%.2f, %.2f) reached, %.3f m away: stopped until a new goal arrives",
                    m_goal.x * m_cellSize, m_goal.y * m_cellSize, goalDist * m_cellSize);
        return;
    }

    ControllerInput in;
    in.pose.x = pose.x / m_cellSize;
    in.pose.y = pose.y / m_cellSize;
    in.pose.theta = pose.theta;
    in.goal = m_goal;
    in.sensor_radius = m_sensorRadius;
    in.nodes = m_nodes;
    buildEdtWindow(in);

    const ControlInput u = m_mppi.computeControl(in);

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = u.v * m_cellSize;
    cmd.angular.z = u.omega;
    m_cmdPub->publish(cmd);

    publishEdt(in);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "pose (%.2f, %.2f, %.2f) goal (%.2f, %.2f) nodes %zu kernels %zu -> v %.2f m/s, w %.2f rad/s",
                         pose.x, pose.y, pose.theta, m_goal.x * m_cellSize, m_goal.y * m_cellSize, m_nodes.size(),
                         m_mppi.kernels.size(), cmd.linear.x, cmd.angular.z);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ReactiveControllerNode>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
