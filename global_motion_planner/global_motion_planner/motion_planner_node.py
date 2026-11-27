import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from messages.msg import RectRegion as RectRegionMsg
from messages.msg import RectRegionArray as RectRegionArrayMsg
from messages.msg import EdtWindow as EdtWindowMsg
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_pose
import math
import numpy as np
import matplotlib.pyplot as plt
from global_motion_planner.ui import Visualizer
from global_motion_planner.graph_generator import RectNodeGraphGenerator
from global_motion_planner.route_planner import DijkstraRoutePlanner

class MotionPlannerNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        is_car = self.declare_parameter('is_car', False).value

        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.fig = None
        self.ax = None
        self.map_image = None

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)

        if is_car:
            self.odom_sub = self.create_subscription(Odometry, '/carla/ego/odometry', self.odom_callback, 10)
            self.distance_threshold = 1.0
            self.min_rect_half_size = 0.5
            self.goal_distance_threshold = 4.0
        else:
            self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
            self.distance_threshold = 0.1
            self.min_rect_half_size = 0.1
            self.goal_distance_threshold = 0.5
        
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.temp_goal_pub = self.create_publisher(PoseStamped, '/temp_goal_point', 10)

        self.current_nodes_pub = self.create_publisher(RectRegionArrayMsg, '/current_nodes', 10)

        # the distance window the controller used on its last tick, drawn in its own window
        self.edt_sub = self.create_subscription(EdtWindowMsg, '/controller_edt', self.edt_callback, 1)
        self.pending_edt = None
        self.edt_fig = None
        self.edt_ax = None
        self.edt_colorbar = None
        self.edt_timer = self.create_timer(0.2, self._draw_edt)

        self.gui_timer = self.create_timer(0.05, self._pump_gui)

        self.route_timer = self.create_timer(0.1, self._update_route)

        self.node_generator = None
        self.route_planner = None

        self.nodes = []
        self.graph = {}

        self.node_patches = []
        self.temp_goal_artists = []

        self.start_artist = None
        self.goal_artist = None

        self.route_nodes = []
        self.temp_goals = []
        self.current_temp_goal_index = 0

        # route node the robot is in, only used to log when it changes
        self.current_node = None

        self.start = None
        self.goal = None

        self._nodes_generated = False

        # True while a route is being followed, False before the first plan and after the final goal
        self.route_active = False

    def map_callback(self, msg):
        self.map = msg
        self.on_map_updated()

        self._generate_nodes()

    def odom_callback(self, msg):
        if self.map is None:
            return

        map_frame = self.map.header.frame_id
        source_frame = msg.header.frame_id

        try:
            transform = self.tf_buffer.lookup_transform(map_frame, source_frame, Time(), timeout=Duration(seconds=0.1))
        except TransformException as exc:
            self.get_logger().warn(f'Could not transform robot pose into {map_frame!r}: {exc}', throttle_duration_sec=2.0)
            return

        pose_map = do_transform_pose(msg.pose.pose, transform)

        self.start = self._world_to_cell(pose_map.position.x, pose_map.position.y)

        self._draw_start()

    def goal_callback(self, msg):
        if self.map is None:
            self.get_logger().warn('Received goal before the map; ignoring.')
            return

        self.goal = self._world_to_cell(msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'New goal at cell {self.goal}. The start at cell {self.start}')

        self._draw_goal()

        self._find_temp_goals()

    # the planner puts cell i at coordinate i (its centre), the map origin is the outer corner of cell 0
    def _world_to_cell(self, world_x, world_y):
        info = self.map.info
        column = (world_x - info.origin.position.x) / info.resolution - 0.5
        row = (world_y - info.origin.position.y) / info.resolution - 0.5
        return (column, row)

    def _cell_to_world(self, column, row):
        info = self.map.info
        world_x = (column + 0.5) * info.resolution + info.origin.position.x
        world_y = (row + 0.5) * info.resolution + info.origin.position.y
        return (world_x, world_y)

    def on_map_updated(self):
        self.get_logger().info("Map updated.")
        if self.map is None:
            return

        grid = self._display_grid_from_msg(self.map)
        columns, rows = grid.shape

        if self.map_image is None:
            plt.ion()
            self.fig, self.ax, _ = Visualizer.visualize_graph(obstacle_grid=grid, nodes=[], block=False, title="Occupancy Grid Map")
            self.map_image = self.ax.images[0]
            self.map_image.set_clim(0.0, 1.0)
        else:
            self.map_image.set_data(grid.T)
            self.map_image.set_extent((-0.5, columns - 0.5, -0.5, rows - 0.5))
            self.ax.set_xlim(-0.5, columns - 0.5)
            self.ax.set_ylim(-0.5, rows - 0.5)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _draw_nodes(self, nodes):
        if self.ax is None:
            return

        for patch in self.node_patches:
            patch.remove()

        self.node_patches = Visualizer.draw_nodes(ax=self.ax, nodes=nodes)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _draw_temp_goals(self):
        if self.ax is None:
            return

        for artist in self.temp_goal_artists:
            artist.remove()

        self.temp_goal_artists = Visualizer.draw_temp_goals(ax=self.ax, temp_goals=self.temp_goals[:-1])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _draw_start(self):
        if self.ax is None or self.start is None:
            return

        if self.start_artist is None:
            self.start_artist = Visualizer.draw_start(ax=self.ax, start=self.start)
        else:
            self.start_artist.set_data([self.start[0]], [self.start[1]])

        self.fig.canvas.draw_idle()

    def _draw_goal(self):
        if self.ax is None or self.goal is None:
            return

        if self.goal_artist is None:
            self.goal_artist = Visualizer.draw_goal(ax=self.ax, goal=self.goal)
        else:
            self.goal_artist.set_data([self.goal[0]], [self.goal[1]])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _generate_nodes(self):
        if self.map is None or self._nodes_generated:
            return

        self._nodes_generated = True

        obstacle_grid = self._obstacle_grid_from_msg(self.map)

        # the generator works in cells
        resolution = self.map.info.resolution
        distance_threshold_cells = self.distance_threshold / resolution
        min_rect_half_size_cells = self.min_rect_half_size / resolution

        self.node_generator = RectNodeGraphGenerator(obstacle_grid=obstacle_grid, alpha=0.99, p_confidence=0.90, gamma=1.05, dst_thr=distance_threshold_cells, robot_radius=min_rect_half_size_cells)
        result = self.node_generator.generate()
        self.nodes = result["nodes"]
        self.graph = result["graph"]

        self.route_planner = DijkstraRoutePlanner(nodes=self.nodes, graph=self.graph)

        self.get_logger().info(f"Generated graph with {len(self.nodes)} nodes (distance threshold {distance_threshold_cells:.1f} cells, min rectangle half size {min_rect_half_size_cells:.1f} cells).")

        self._draw_nodes(self.nodes)

    def _obstacle_grid_from_msg(self, msg, occupied_threshold=50):
        raw = self._raw_grid_from_msg(msg)
        return (raw >= occupied_threshold) | (raw < 0)

    def _display_grid_from_msg(self, msg):
        raw = self._raw_grid_from_msg(msg)

        display = np.full(raw.shape, 0.5, dtype=float)
        known = raw >= 0
        display[known] = raw[known] / 100.0
        return display

    def _raw_grid_from_msg(self, msg):
        width = msg.info.width
        height = msg.info.height
        return np.asarray(msg.data, dtype=np.int8).reshape(height, width).T

    def _find_temp_goals(self):
        if self.route_planner is None or not self.nodes:
            self.get_logger().warn('No node graph yet; cannot plan a route.')
            return

        if self.start is None or self.goal is None:
            self.get_logger().warn('Need both a robot pose (/odom) and a goal to plan.')
            return

        try:
            self.route_planner.find_route(start=self.start, goal=self.goal)
        except (ValueError, RuntimeError) as exc:
            self.get_logger().warn(f'Route planning failed: {exc}')
            self.route_active = False
            return

        self._set_pruned_route()
        self.route_active = True

        self.get_logger().info(f'Route found: {len(self.route_nodes)} nodes, {len(self.temp_goals)} temp goals.')

    def _set_pruned_route(self):
        temp_goals = self.route_planner.get_pruned_temp_goal_points()
        self._set_route(temp_goals, self.route_planner.pruned_route_node_ids)

    def _set_route(self, temp_goals, route_node_ids):
        self.route_nodes = [self.nodes[node_id] for node_id in route_node_ids]
        self.temp_goals = temp_goals
        self.current_temp_goal_index = 0

        self._draw_nodes(self.route_nodes)
        self._draw_temp_goals()

        # the nodes for the new temp goal go out on the next route tick
        self._publish_temp_goal(self.current_temp_goal_index)

    def _update_route(self):
        # runs on a timer, so the node checks below do not run on every odom message
        if not self.route_active or self.start is None:
            return

        position = (float(self.start[0]), float(self.start[1]))

        if not any(node.contains(position) for node in self.route_nodes):
            self._replan_from(position)

        self._check_direct_route(position)

        self._update_temp_goal_progress()

        # after the temp goal hand-over, so the nodes always lead to the temp goal the controller has
        self._publish_current_nodes(position)

    def _replan_from(self, position):
        # the robot left every route node: plan again from where it is now
        try:
            self.route_planner.update_route(position)
        except ValueError:
            # outside every node there is nothing to plan from, so keep the current temp goals
            self.get_logger().warn(f'Robot at cell ({position[0]:.1f}, {position[1]:.1f}) is outside every node; keeping the current route.', throttle_duration_sec=2.0)
            return
        except RuntimeError as exc:
            self.get_logger().warn(f'Route update failed: {exc}', throttle_duration_sec=2.0)
            return

        self._set_pruned_route()

        self.get_logger().info(f'Robot left the route at cell ({position[0]:.1f}, {position[1]:.1f}); new route has {len(self.route_nodes)} nodes and {len(self.temp_goals)} temp goals.')

    def _check_direct_route(self, position):
        # the straight line from the robot to the goal runs through nodes only: drive to the goal directly
        direct_node_ids = self.route_planner.get_direct_route_to_goal(position)

        if direct_node_ids is None:
            return

        # already heading for the final goal: only the nodes around the line change, nothing new to publish
        if self.current_temp_goal_index >= len(self.temp_goals) - 1:
            self.route_nodes = [self.nodes[node_id] for node_id in direct_node_ids]
            return

        self._set_route([(float(self.goal[0]), float(self.goal[1]))], direct_node_ids)

        self.get_logger().info(f'Direct line to the goal from cell ({position[0]:.1f}, {position[1]:.1f}); skipping the remaining temp goals.')

    def _temp_goal_yaw(self, index, world_x, world_y):
        if index + 1 < len(self.temp_goals):
            next_x, next_y = self._cell_to_world(*self.temp_goals[index + 1])
            return math.atan2(next_y - world_y, next_x - world_x)

        # last temp goal: keep the heading it is approached with, from the previous temp goal or the robot
        if index > 0:
            previous = self.temp_goals[index - 1]
        elif self.start is not None:
            previous = self.start
        else:
            return 0.0

        prev_x, prev_y = self._cell_to_world(*previous)
        return math.atan2(world_y - prev_y, world_x - prev_x)

    def _publish_temp_goal(self, index):
        if self.map is None or not (0 <= index < len(self.temp_goals)):
            return

        column, row = self.temp_goals[index]
        world_x, world_y = self._cell_to_world(column, row)

        yaw = self._temp_goal_yaw(index, world_x, world_y)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map.header.frame_id
        msg.pose.position.x = float(world_x)
        msg.pose.position.y = float(world_y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.temp_goal_pub.publish(msg)
        self.get_logger().info(
            f'Published temp goal {index + 1}/{len(self.temp_goals)} '
            f'at world ({world_x:.2f}, {world_y:.2f}).'
        )

    def _update_temp_goal_progress(self):
        if not self.temp_goals or self.current_temp_goal_index >= len(self.temp_goals):
            return

        robot_x, robot_y = self._cell_to_world(*self.start)

        column, row = self.temp_goals[self.current_temp_goal_index]
        goal_x, goal_y = self._cell_to_world(column, row)

        distance = math.hypot(goal_x - robot_x, goal_y - robot_y)

        if distance > self.goal_distance_threshold:
            return

        # temp goal reached: send the next one (the last one is the final goal)
        if self.current_temp_goal_index < len(self.temp_goals) - 1:
            self.current_temp_goal_index += 1
            self._publish_temp_goal(self.current_temp_goal_index)
        else:
            # stop route supervision, the controller keeps driving to the final goal it already has
            self.route_active = False
            self.get_logger().info('Final goal reached; route complete.')

    @staticmethod
    def _distance_to_node(node, point):
        # 0 when the point is inside the node, otherwise its distance to the node's rectangle
        displacement = np.array([float(point[0]), float(point[1])]) - node.center
        outside_1 = max(abs(float(displacement @ node.axis_1)) - node.half_width, 0.0)
        outside_2 = max(abs(float(displacement @ node.axis_2)) - node.half_height, 0.0)

        return math.hypot(outside_1, outside_2)

    def _nodes_to_temp_goal(self, position):
        # the route node the robot is in, then the route nodes after it up to the one holding the temp goal
        # empty when the robot is outside every route node
        containing = [index for index, node in enumerate(self.route_nodes) if node.contains(position)]

        if not containing or not self.temp_goals:
            return []

        # in an overlap take the node furthest along the route, so progress is not pulled backwards
        current_index = containing[-1]

        temp_goal = self.temp_goals[self.current_temp_goal_index]

        # first node holding the temp goal, or the closest one when none holds it
        distances = [self._distance_to_node(node, temp_goal) for node in self.route_nodes[current_index:]]
        inside = [offset for offset, distance in enumerate(distances) if distance == 0.0]
        goal_index = current_index + (inside[0] if inside else int(np.argmin(distances)))

        return self.route_nodes[current_index:goal_index + 1]

    def _node_msg(self, node):
        # cells to meters in the map frame, the controller works in meters
        resolution = self.map.info.resolution
        center_x, center_y = self._cell_to_world(node.center_x, node.center_y)

        msg = RectRegionMsg()
        msg.center_x = float(center_x)
        msg.center_y = float(center_y)
        msg.theta = float(node.theta)
        msg.half_width = float(node.half_width * resolution)
        msg.half_height = float(node.half_height * resolution)
        return msg

    def _publish_current_nodes(self, position):
        nodes = self._nodes_to_temp_goal(position)

        msg = RectRegionArrayMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map.header.frame_id
        msg.regions = [self._node_msg(node) for node in nodes]

        # sent every tick like the reference loop, so a controller that starts late or gets its goal late still has the nodes
        self.current_nodes_pub.publish(msg)

        current = nodes[0] if nodes else None

        if current is not self.current_node:
            self.current_node = current

            if current is None:
                self.get_logger().info('Robot is outside every route node; sending no nodes.')
            else:
                self.get_logger().info(f'Robot entered node at cell ({current.center_x:.1f}, {current.center_y:.1f}); sending {len(nodes)} nodes up to the temp goal.')

    def edt_callback(self, msg):
        # drawn by _draw_edt at a lower rate, only the latest window matters
        self.pending_edt = msg

    def _draw_edt(self):
        msg = self.pending_edt
        self.pending_edt = None

        if msg is None or self.map is None or msg.size_x == 0 or msg.size_y == 0:
            return

        distances = np.asarray(msg.distances, dtype=float).reshape(msg.size_x, msg.size_y)

        # the Ackermann controller sends its rectangle: the image is then the plain distance to obstacles with the rectangle
        # drawn on top; with a circle it is the distance from the robot's edge, not its centre (< 0 means collision)
        rectangle = msg.footprint_half_width > 0.0
        radius = 0.0 if rectangle else msg.robot_radius
        clearance = distances - radius

        # window corners in map cells, cell [i][j] is centred at origin + (i, j) * resolution
        half = 0.5 * msg.resolution
        x_min, y_min = self._world_to_cell(msg.origin_x - half, msg.origin_y - half)
        x_max, y_max = self._world_to_cell(msg.origin_x + (msg.size_x - 0.5) * msg.resolution, msg.origin_y + (msg.size_y - 0.5) * msg.resolution)

        robot_x, robot_y = self._world_to_cell(msg.robot_x, msg.robot_y)
        cells_per_meter = 1.0 / self.map.info.resolution
        robot_radius = msg.robot_radius * cells_per_meter

        # the gradient is unitless (meters per meter)
        grad_x, grad_y = np.gradient(distances / msg.resolution)

        if rectangle:
            footprint = (msg.footprint_front * cells_per_meter, msg.footprint_rear * cells_per_meter, msg.footprint_half_width * cells_per_meter)
            # the normal the controller uses: the gradient at the outline point closest to an obstacle
            i = min(max(int(round((msg.closest_x - msg.origin_x) / msg.resolution)), 0), msg.size_x - 1)
            j = min(max(int(round((msg.closest_y - msg.origin_y) / msg.resolution)), 0), msg.size_y - 1)
            gradient_origin = self._world_to_cell(msg.closest_x, msg.closest_y)
            gradient = (grad_x[i, j], grad_y[i, j])
            label = f'footprint clearance = {msg.footprint_clearance:.2f} m'
        else:
            # the window is centred on the robot's cell
            center_x, center_y = msg.size_x // 2, msg.size_y // 2
            footprint = None
            gradient_origin = None
            gradient = (grad_x[center_x, center_y], grad_y[center_x, center_y])
            label = f'clearance = {clearance[center_x, center_y]:.2f} m'

        if self.edt_fig is None:
            plt.ion()
            self.edt_fig, self.edt_ax = plt.subplots(figsize=(7, 6))
            plt.show(block=False)

        # fixed colour scale, from touching an obstacle to free up to the sensor radius
        image = Visualizer.draw_edt(
            ax=self.edt_ax,
            clearance=clearance,
            extent=(x_min, x_max, y_min, y_max),
            clim=(-radius, msg.max_distance - radius),
            robot_pose=(robot_x, robot_y, msg.robot_theta),
            robot_radius=robot_radius,
            gradient=gradient,
            footprint=footprint,
            gradient_origin=gradient_origin,
        )

        self.edt_ax.text(0.02, 0.02, label, transform=self.edt_ax.transAxes, bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'})

        if self.edt_colorbar is None:
            colorbar_label = 'Distance to obstacle (m)' if rectangle else 'Clearance (m)'
            self.edt_colorbar = self.edt_fig.colorbar(image, ax=self.edt_ax, label=colorbar_label, fraction=0.046, pad=0.04)
        else:
            self.edt_colorbar.update_normal(image)

        self.edt_fig.canvas.draw_idle()

    def _pump_gui(self):
        if self.fig is not None:
            self.fig.canvas.flush_events()

        if self.edt_fig is not None:
            self.edt_fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down motion planner...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
