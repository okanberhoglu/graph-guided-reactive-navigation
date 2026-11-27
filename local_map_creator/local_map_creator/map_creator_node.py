import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from tf2_ros import Buffer, TransformListener, TransformException

class MapCreatorNode(Node):
    def __init__(self):
        super().__init__('map_creator_node')

        is_car = self.declare_parameter('is_car', False).value

        self.ocp_publisher_ = self.create_publisher(OccupancyGrid, '/local_map', 10)

        if is_car:
            self.scan_subscriber_ = self.create_subscription(LaserScan, '/carla/ego/scan', self.laser_scan_callback, qos_profile_sensor_data)
            self.grid_resolution = 0.1
            self.grid_width = 200
            self.grid_height = 200
        else:
            self.scan_subscriber_ = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, qos_profile_sensor_data)
            self.grid_resolution = 0.1
            self.grid_width = 80
            self.grid_height = 80
        
        self.map_frame = 'map'
        self.robot_frame = 'base_link'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # rolling local map size in meters: 20 m x 20 m for the car, 8 m x 8 m otherwise
        self.grid_width_m = self.grid_width * self.grid_resolution
        self.grid_height_m = self.grid_height * self.grid_resolution

        self.scan_queue = deque(maxlen=10)

        self.processing_timer = self.create_timer(0.01, self.process_scan_queue)

        self.get_logger().info('Map-frame rolling occupancy-grid generator initialized')

    def laser_scan_callback(self, msg: LaserScan):
        self.scan_queue.append(msg)

    def process_scan_queue(self):
        if not self.scan_queue:
            return

        selected_scan = None
        selected_time = None

        # Choose the newest scan whose transforms are available.
        for scan in reversed(self.scan_queue):
            scan_time = Time.from_msg(scan.header.stamp)

            scan_transform_available = self.tf_buffer.can_transform(self.map_frame, scan.header.frame_id, scan_time)

            robot_transform_available = self.tf_buffer.can_transform(self.map_frame, self.robot_frame, scan_time)

            if scan_transform_available and robot_transform_available:
                selected_scan = scan
                selected_time = scan_time
                break

        if selected_scan is None:
            return

        self.scan_queue.clear()

        self.generate_local_map(selected_scan, selected_time)

    def generate_local_map(self, scan: LaserScan, scan_time: Time):
        try:
            scan_to_map = self.tf_buffer.lookup_transform(self.map_frame, scan.header.frame_id, scan_time)

            robot_to_map = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame, scan_time)

        except TransformException as exception:
            self.get_logger().warning(f'Could not transform scan into {self.map_frame}: {exception}')
            return

        instant_grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)

        robot_x_map = robot_to_map.transform.translation.x
        robot_y_map = robot_to_map.transform.translation.y

        origin_x = robot_x_map - self.grid_width_m / 2.0
        origin_y = robot_y_map - self.grid_height_m / 2.0

        laser_x_map = scan_to_map.transform.translation.x
        laser_y_map = scan_to_map.transform.translation.y

        laser_yaw_map = self.quaternion_to_yaw(scan_to_map.transform.rotation)

        cos_laser_yaw = math.cos(laser_yaw_map)
        sin_laser_yaw = math.sin(laser_yaw_map)

        laser_grid_x, laser_grid_y = self.map_to_grid(laser_x_map, laser_y_map, origin_x, origin_y)

        if not self.is_valid_grid_cell(laser_grid_x, laser_grid_y):
            self.get_logger().warning('Laser origin is outside the generated local map')
            return

        for i, measured_range in enumerate(scan.ranges):
            if math.isnan(measured_range):
                continue

            angle_scan = scan.angle_min + i * scan.angle_increment

            has_obstacle_hit = (math.isfinite(measured_range) and scan.range_min <= measured_range < scan.range_max)

            if has_obstacle_hit:
                ray_range = measured_range
            else:
                ray_range = scan.range_max

            endpoint_x_scan = ray_range * math.cos(angle_scan)
            endpoint_y_scan = ray_range * math.sin(angle_scan)

            endpoint_x_map = (laser_x_map + cos_laser_yaw * endpoint_x_scan - sin_laser_yaw * endpoint_y_scan)

            endpoint_y_map = (laser_y_map + sin_laser_yaw * endpoint_x_scan + cos_laser_yaw * endpoint_y_scan)

            endpoint_grid_x, endpoint_grid_y = self.map_to_grid(endpoint_x_map, endpoint_y_map, origin_x, origin_y)

            self.trace_ray_instant(instant_grid, laser_grid_x, laser_grid_y, endpoint_grid_x, endpoint_grid_y)

            if (has_obstacle_hit and self.is_valid_grid_cell(endpoint_grid_x, endpoint_grid_y)):
                instant_grid[endpoint_grid_y, endpoint_grid_x] = 100

        occupancy_grid_msg = self.create_occupancy_grid_map(instant_grid, origin_x, origin_y, scan.header.stamp)

        self.ocp_publisher_.publish(occupancy_grid_msg)

    def map_to_grid(self, x_map: float, y_map: float, origin_x: float, origin_y: float):
        grid_x = math.floor((x_map - origin_x) / self.grid_resolution)

        grid_y = math.floor((y_map - origin_y) / self.grid_resolution)

        return int(grid_x), int(grid_y)

    def is_valid_grid_cell(self, grid_x: int, grid_y: int):
        return (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height)

    def trace_ray_instant(self, grid: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        """
        Bresenham ray tracing.

        The starting point is the laser origin in the rolling grid. The ray
        can terminate outside the grid; tracing stops when it leaves the map.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x = x0
        y = y0

        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1

        error = dx - dy

        dx_twice = 2 * dx
        dy_twice = 2 * dy

        maximum_steps = 1 + dx + dy

        for _ in range(maximum_steps):
            if not self.is_valid_grid_cell(x, y):
                break

            if x == x1 and y == y1:
                break

            grid[y, x] = 0

            if error > 0:
                x += x_inc
                error -= dy_twice
            else:
                y += y_inc
                error += dx_twice

    def create_occupancy_grid_map(self,grid: np.ndarray,origin_x: float,origin_y: float,stamp):
        occupancy_grid = OccupancyGrid()

        occupancy_grid.header.stamp = stamp
        occupancy_grid.header.frame_id = self.map_frame

        occupancy_grid.info.resolution = self.grid_resolution
        occupancy_grid.info.width = self.grid_width
        occupancy_grid.info.height = self.grid_height

        occupancy_grid.info.origin = Pose(
            position=Point(
                x=origin_x,
                y=origin_y,
                z=0.0
            ),
            orientation=Quaternion(
                x=0.0,
                y=0.0,
                z=0.0,
                w=1.0
            )
        )

        occupancy_grid.data = grid.flatten().tolist()

        return occupancy_grid

    @staticmethod
    def quaternion_to_yaw(quaternion) -> float:
        sin_yaw = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cos_yaw = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(sin_yaw, cos_yaw)


def main(args=None):
    rclpy.init(args=args)
    node = MapCreatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down occupancy-grid generator')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()