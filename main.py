import argparse
import sys
import numpy as np
import math
from matplotlib.quiver import Quiver
from utils.utils import MapCreator, Utils, CounterflowPedestrians, CrossflowPedestrians
from global_planner.graph_generator import RectNodeGraphGenerator
from global_planner.route_planner import DijkstraRoutePlanner
from model.robot_config import RobotConfig, CarConfig
from utils.ui import Visualizer
from utils.udp_connection import UDPConnection
from model.robot_model import ackerman_dd_step, dd_step

DEFAULT_MODE = "robot"
MODES = ("robot", "car", "social_cross", "social_counter", "global")

MAP_MODES = ("robot", "car")

DEFAULT_MAP = 1


def _available_maps():
    """Map numbers exposed by MapCreator as get_map<N>()."""
    prefix = "get_map"
    numbers = []
    for name in dir(MapCreator):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            numbers.append(int(name[len(prefix):]))
    return tuple(sorted(numbers))


MAP_NUMBERS = _available_maps()

def get_global_route(map_number=DEFAULT_MAP, min_node_half_size=RobotConfig.robot_radius, dst_thr=2.0, is_car = False):
    obstacle_grid, start, goal = get_map(map_number)

    if is_car:
        map_scale = 4
        obstacle_grid = np.kron(obstacle_grid, np.ones((map_scale, map_scale), dtype=bool))
        start = (map_scale * start[0] + (map_scale - 1) / 2, map_scale * start[1] + (map_scale - 1) / 2)
        goal = (map_scale * goal[0] + (map_scale - 1) / 2, map_scale * goal[1] + (map_scale - 1) / 2)

    waypoints, route_nodes, nodes, route_planner = plan_route(obstacle_grid, start, goal, min_node_half_size=min_node_half_size, dst_thr=dst_thr)

    return waypoints, route_nodes, nodes, obstacle_grid, start, goal, route_planner

def plan_route(obstacle_grid, start, goal, min_node_half_size, dst_thr):
    rectNodegenerator = RectNodeGraphGenerator(obstacle_grid=obstacle_grid, alpha=0.99, p_confidence=0.90, gamma=1.05, dst_thr=dst_thr, robot_radius=min_node_half_size)

    result = rectNodegenerator.generate()

    nodes = result['nodes']
    graph = result['graph']

    route_planner = DijkstraRoutePlanner(nodes=nodes, graph=graph)
    route_planner.find_route(start=start, goal=goal)

    waypoints = route_planner.get_pruned_temp_goal_points()
    route_nodes = [nodes[node_id] for node_id in route_planner.pruned_route_node_ids]

    return waypoints, route_nodes, nodes, route_planner

def local_free_mask(obstacle_grid, robot_pose, sensor_radius, visible_mask):
    # local map: 0 = free, 1 = obstacle, 0.5 = unknown (not visible / out of range)
    # same [x, y] layout as extract_local_occupancy
    local_map = Visualizer.extract_local_map(obstacle_grid, robot_pose, sensor_radius, visible_mask).T

    # only known-free cells are free; the controller counts the unknown cells inside the sensor circle (behind
    # obstacles) as obstacles, not the ones outside it, so the sensor boundary is no obstacle
    return local_map == 0.0

def global_sensor_radius(obstacle_grid, udp):
    # the whole map is known: the sensor range is the map diagonal, so the window around the robot covers the whole map
    # from any cell of it; the window's free mask (one uint8 per cell) must fit in one UDP request
    columns, rows = obstacle_grid.shape
    sensor_radius = float(math.ceil(math.hypot(columns, rows)))
    window = 2 * int(sensor_radius) + 1

    if udp.request_header.size + window * window + udp.count.size > udp.max_datagram:
        raise ValueError(f"a {columns} x {rows} map needs a {window} x {window} sensor window, which does not fit in one UDP request")

    return sensor_radius

def global_free_mask(obstacle_grid, robot_pose, sensor_radius):
    # free mask of the sensor window with nothing occluded: every map cell is known, the cells outside the map are not
    # free, so the controller counts them as obstacles; same [x, y] layout as local_free_mask
    local_map = Visualizer.extract_local_map(obstacle_grid, robot_pose, sensor_radius, visible_mask=None).T

    return local_map == 0.0

def show_controller_edt(robot_viz, controller_edt, obstacle_grid):
    # the controller's EDT window reaches far past the map: draw only the map, with the colours spread over the
    # clearances inside the map instead of up to the (map diagonal) sensor range
    first_draw = robot_viz.edt_colorbar is None
    robot_viz.update_controller_edt(controller_edt)
    if controller_edt is None:
        return

    columns, rows = obstacle_grid.shape
    origin_x, origin_y = (int(round(origin)) for origin in controller_edt["origin"])
    in_map = controller_edt["distances"][max(0, -origin_x):columns - origin_x, max(0, -origin_y):rows - origin_y]
    radius = controller_edt["robot_radius"]

    if in_map.size:
        image = robot_viz.ax_edt.images[0]
        image.set_clim(-radius, in_map.max() - radius)
        robot_viz.edt_colorbar.update_normal(image)

    # the gradient arrow is sized to the window, draw a unit gradient one robot radius long instead
    gradient_arrow = next(artist for artist in robot_viz.ax_edt.collections if isinstance(artist, Quiver))
    grad_x, grad_y = float(gradient_arrow.U[0]), float(gradient_arrow.V[0])
    norm = math.hypot(grad_x, grad_y)
    if norm > 1.0:
        gradient_arrow.set_UVC(grad_x / norm, grad_y / norm)
    gradient_arrow.scale = 1.0 / radius

    robot_viz.ax_edt.set_xlim(-0.5, columns - 0.5)
    robot_viz.ax_edt.set_ylim(-0.5, rows - 0.5)

    # lay the figure out again once the EDT view has its colorbar and map limits, otherwise its labels run under the map
    if first_draw:
        robot_viz.fig.tight_layout()

def update_temp_goals(route_planner, robot_pose, pruned=True):
    position = (float(robot_pose[0]), float(robot_pose[1]))

    # outside every node there is nothing to plan from: None tells the caller to keep the current temp goals
    try:
        route_node_ids = route_planner.update_route(position)
    except ValueError:
        return None

    if pruned:
        temp_goals = route_planner.get_pruned_temp_goal_points()
        # the shortcuts can run through nodes off the Dijkstra route, so use the pruned node chain
        route_node_ids = route_planner.pruned_route_node_ids
    else:
        temp_goals = route_planner.get_temp_goal_points()

    route_nodes = [route_planner.nodes[node_id] for node_id in route_node_ids]

    return route_nodes, temp_goals

def distance_to_node(node, point):
    # 0 when the point is inside the node, otherwise its distance to the node's rectangle
    displacement = np.array([float(point[0]), float(point[1])]) - node.center
    outside_1 = max(abs(float(displacement @ node.axis_1)) - node.half_width, 0.0)
    outside_2 = max(abs(float(displacement @ node.axis_2)) - node.half_height, 0.0)

    return math.hypot(outside_1, outside_2)

def get_nodes_to_temp_goal(robot_pose, route_nodes, temp_goal):
    
    position = (float(robot_pose[0]), float(robot_pose[1]))

    containing = [index for index, node in enumerate(route_nodes) if node.contains(position)]
    if not containing:
        return None

    current_index = containing[-1]

    distances = [distance_to_node(node, temp_goal) for node in route_nodes[current_index:]]
    inside = [offset for offset, distance in enumerate(distances) if distance == 0.0]
    goal_index = current_index + (inside[0] if inside else int(np.argmin(distances)))

    return route_nodes[current_index], route_nodes[current_index + 1:goal_index + 1]


def get_map(map_number):
    return getattr(MapCreator, f"get_map{map_number}")()


def robot_controller_step(udp, robot_pose, temp_goal, sensor_radius, free_mask, nodes_to_send, robot_viz, is_final):
    u, field, kernels, controller_edt = udp.controller_messaging(robot_pose, temp_goal, sensor_radius, free_mask, nodes_to_send, robot_viz, is_final=is_final)

    return u, field, kernels, udp.kernel_gamma, controller_edt, False


def car_controller_step(udp, robot_pose, temp_goal, sensor_radius, free_mask, nodes_to_send, robot_viz, is_final):
    u, field, kernels, temp_goal_reached, controller_edt = udp.car_controller_messaging(robot_pose, temp_goal, sensor_radius, free_mask, nodes_to_send, robot_viz)

    return u, field, kernels, udp.car_kernel_gamma, controller_edt, temp_goal_reached


def navigate(robot_viz, obstacle_grid, route_planner, nodes, route_nodes, temp_goals, goal, robot_pose, sensor_radius, temp_goal_switch_dist, controller_step, motion_step, robot_radius=None, footprint=None, pedestrians=None, dt=0.1):
    # pedestrians (robot only): moving obstacles the route does not know about, the robot only sees them through its sensor
    i = 0

    path = [(robot_pose[0], robot_pose[1])]

    # every route node used so far, starting with the route visualize_graph drew
    drawn_nodes = list(route_nodes)

    with UDPConnection() as udp:
        while True:
            # what the sensor sees: the static map, with every pedestrian stamped in where it is now
            sensed_grid = obstacle_grid if pedestrians is None else pedestrians.grid()

            visible_mask = Visualizer.visibility_mask(sensed_grid, robot_pose, sensor_radius)

            free_mask = local_free_mask(sensed_grid, robot_pose, sensor_radius, visible_mask)

            robot_viz.obstacle_grid = sensed_grid
            robot_viz.visible_mask = visible_mask

            position = (float(robot_pose[0]), float(robot_pose[1]))

            if not any(node.contains(position) for node in route_nodes):
                updated = update_temp_goals(route_planner, robot_pose, pruned=True)

                if updated is not None:
                    route_nodes, temp_goals = updated
                    i = 0
                    print(f"[Py] robot left the route at ({robot_pose[0]:.1f}, {robot_pose[1]:.1f}), new route has {len(route_nodes)} nodes and {len(temp_goals)} temp goals", flush=True)

            direct_node_ids = route_planner.get_direct_route_to_goal(position=position)

            if direct_node_ids is not None:
                route_nodes = [nodes[k] for k in direct_node_ids]
                temp_goals = [(float(goal[0]), float(goal[1]))]
                i = 0
                print(f"[Py] direct line to the goal from ({position[0]:.1f}, {position[1]:.1f})", flush=True)

            # a reroute or a direct line to the goal adds its new nodes to the drawing, the old ones stay
            new_nodes = [node for node in route_nodes if node not in drawn_nodes]

            if new_nodes:
                drawn_nodes += new_nodes
                robot_viz.update_nodes(drawn_nodes)

            temp_goal = temp_goals[i]

            is_final = i == len(temp_goals) - 1

            temp_nodes = get_nodes_to_temp_goal(robot_pose=robot_pose, temp_goal=temp_goal, route_nodes=route_nodes)

            nodes_to_send = [] if temp_nodes is None else [temp_nodes[0]] + temp_nodes[1]

            u, field, kernels, kernel_gamma, controller_edt, temp_goal_reached = controller_step(udp, robot_pose, temp_goal, sensor_radius, free_mask, nodes_to_send, robot_viz, is_final)

            robot_viz.update_vector_field(field[:, :2], field[:, 2:])
            robot_viz.update_kernels(kernels[:, :2], kernels[:, 2:], gamma=kernel_gamma)

            robot_pose = motion_step(robot_pose, u, dt)
            path.append((robot_pose[0], robot_pose[1]))

            if pedestrians is not None:
                pedestrians.step(dt, robot_pose, robot_radius)

            if footprint is None:
                collided = Utils.check_collision(obstacle_grid, robot_pose, robot_radius)
            else:
                collided = Utils.check_car_collision(obstacle_grid, robot_pose, footprint)

            # pedestrians are checked by distance, the ones stamped in sensed_grid are from before this step
            hit_pedestrian = pedestrians is not None and pedestrians.min_distance(robot_pose) < robot_radius + pedestrians.radius

            if collided or hit_pedestrian:
                obstacle = "a pedestrian" if hit_pedestrian else "an obstacle"
                print(f"[Py] collision with {obstacle} at ({robot_pose[0]:.1f}, {robot_pose[1]:.1f}) after {len(path) - 1} ticks, stopping", flush=True)
                udp.controller_stop()
                return

            temp_goal_dist = Utils.calc_dist((robot_pose[0], robot_pose[1]), temp_goal)
            switch_temp_goal = temp_goal_dist < temp_goal_switch_dist

            if switch_temp_goal and not is_final:
                i += 1

            if pedestrians is not None:
                robot_viz.update_moving_obstacles(pedestrians.positions, pedestrians.radius, pedestrians.velocities)

            robot_viz.update_trajectory(path)
            robot_viz.update(robot_pose=robot_pose, pause=0.1)
            robot_viz.update_temp_goals(temp_goals)
            robot_viz.update_controller_edt(controller_edt)

            if footprint is None:
                goal_reached = Utils.calc_dist((robot_pose[0], robot_pose[1]), goal) < robot_radius
            else:
                goal_reached = temp_goal_reached and is_final

            if goal_reached:
                print(f"[Py] goal reached after {len(path) - 1} ticks", flush=True)
                udp.controller_stop()


def run_robot(map_number=DEFAULT_MAP):
    print(f"[mode] robot (map {map_number})")
    min_node_half_size = RobotConfig.robot_radius
    waypoints, route_nodes, nodes, obstacle_grid, start, goal, route_planner = get_global_route(map_number=map_number, min_node_half_size=min_node_half_size)
    robot_pose = [start[0], start[1], 0.0]
    robot_radius = RobotConfig.robot_radius
    sensor_radius = RobotConfig.sensor_radius
    fig, ax, robot_viz = Visualizer.visualize_graph(obstacle_grid=obstacle_grid, nodes=route_nodes, temp_goals=waypoints, goal=goal, start=start, robot_pose=robot_pose, robot_radius=robot_radius, sensor_radius=sensor_radius, block=False, save_global_map=False, save_local_map=False, save_edt_map=False, footprint=None)

    navigate(
        robot_viz=robot_viz,
        obstacle_grid=obstacle_grid,
        route_planner=route_planner,
        nodes=nodes,
        route_nodes=route_nodes,
        temp_goals=waypoints,
        goal=goal,
        robot_pose=robot_pose,
        sensor_radius=sensor_radius,
        temp_goal_switch_dist=robot_radius * 3,
        controller_step=robot_controller_step,
        motion_step=dd_step,
        robot_radius=robot_radius,
    )


def run_car(map_number=DEFAULT_MAP):
    print("[mode] car")
    dst_thr = 10.0
    min_node_half_size = 5.0

    waypoints, route_nodes, nodes, obstacle_grid, start, goal, route_planner = get_global_route(map_number=map_number, min_node_half_size=min_node_half_size, dst_thr=dst_thr, is_car=True)

    footprint = (CarConfig.car_front, CarConfig.car_rear, CarConfig.car_half_width)
    robot_pose = [start[0], start[1], 0.0, 0.0]
    sensor_radius = CarConfig.sensor_radius
    fig, ax, robot_viz = Visualizer.visualize_graph(obstacle_grid=obstacle_grid, nodes=route_nodes, temp_goals=waypoints, goal=goal, start=start, robot_pose=robot_pose, robot_radius=None, sensor_radius=sensor_radius, block=False, save_global_map=False, save_local_map=False, save_edt_map=False, footprint=footprint)

    navigate(
        robot_viz=robot_viz,
        obstacle_grid=obstacle_grid,
        route_planner=route_planner,
        nodes=nodes,
        route_nodes=route_nodes,
        temp_goals=waypoints,
        goal=goal,
        robot_pose=robot_pose,
        sensor_radius=sensor_radius,
        temp_goal_switch_dist=40.0,
        controller_step=car_controller_step,
        motion_step=ackerman_dd_step,
        footprint=footprint,
    )


def run_social(obstacle_grid, start, goal, pedestrians):
    # robot mode with pedestrians: the route comes from the static map, the pedestrians are only seen by the sensor
    robot_radius = RobotConfig.robot_radius
    sensor_radius = RobotConfig.sensor_radius

    waypoints, route_nodes, nodes, route_planner = plan_route(obstacle_grid, start, goal, min_node_half_size=robot_radius, dst_thr=0.0)

    robot_pose = [start[0], start[1], 0.0]
    fig, ax, robot_viz = Visualizer.visualize_graph(obstacle_grid=obstacle_grid, nodes=route_nodes, temp_goals=waypoints, goal=goal, start=start, robot_pose=robot_pose, robot_radius=robot_radius, sensor_radius=sensor_radius, block=False, save_global_map=False, save_local_map=False, save_edt_map=False, footprint=None)

    navigate(
        robot_viz=robot_viz,
        obstacle_grid=obstacle_grid,
        route_planner=route_planner,
        nodes=nodes,
        route_nodes=route_nodes,
        temp_goals=waypoints,
        goal=goal,
        robot_pose=robot_pose,
        sensor_radius=sensor_radius,
        temp_goal_switch_dist=robot_radius * 2.5,
        controller_step=robot_controller_step,
        motion_step=dd_step,
        robot_radius=robot_radius,
        pedestrians=pedestrians,
    )


def run_social_cross():
    print("[mode] social_cross")
  
    corridor_width = 15
   
    obstacle_grid = MapCreator.get_counterflow_map(shape=(120, corridor_width + 2), corridor_rows=(1, corridor_width))

    centre_y = (corridor_width + 1) // 2
    start = (8, centre_y)
    goal = (112, centre_y)

    pedestrian_speed = 1.0
    bottom, top = 3.0, corridor_width - 2.0
    pedestrians = CrossflowPedestrians(obstacle_grid,
                                       positions=[(40.0, top), (55.0, centre_y), (70.0, top), (85.0, centre_y)],
                                       velocities=[(0.0, -pedestrian_speed), (0.0, pedestrian_speed), (0.0, -pedestrian_speed), (0.0, pedestrian_speed)],
                                       radius=2.0,
                                       y_bounds=[(bottom, top)] * 4)

    run_social(obstacle_grid, start, goal, pedestrians)


def run_social_counter():
    print("[mode] social_counter")
    
    obstacle_grid = MapCreator.get_counterflow_map(shape=(120, 35), corridor_rows=(1, 33))

    start = (8, 17)
    goal = (112, 17)

    pedestrian_speed = 2.5
    pedestrians = CounterflowPedestrians(obstacle_grid,
                                         positions=[(70.0, 13.0), (70.0, 21.0), (88.0, 13.0), (88.0, 21.0)],
                                         velocities=[(-pedestrian_speed, 0.0)] * 4,
                                         radius=2.0,
                                         x_bounds=[(30.0, 74.0), (30.0, 74.0), (82.0, 104.0), (82.0, 104.0)])

    run_social(obstacle_grid, start, goal, pedestrians)

def run_global():
    print("[mode] global")
    # U trap with the whole map known and no global planner: the controller alone drives the robot to the goal, with no
    # temp goals and no route nodes; there is no sensing either, it gets the free mask of the whole map every tick
    obstacle_grid, start, goal = MapCreator.get_u_trap_map()

    robot_pose = [start[0], start[1], 0.0]
    robot_radius = RobotConfig.robot_radius

    udp = UDPConnection()
    sensor_radius = global_sensor_radius(obstacle_grid, udp)

    # the sensor views are hidden below, so the visualizer only gets the robot's usual sensor range for them
    fig, ax, robot_viz = Visualizer.visualize_graph(obstacle_grid=obstacle_grid, nodes=None, temp_goals=None, goal=goal, start=start, robot_pose=robot_pose, robot_radius=robot_radius, sensor_radius=RobotConfig.sensor_radius, block=False, save_global_map=False, save_local_map=False, save_edt_map=False, footprint=None)

    # no sensor: hide the sensor range, drop the local occupancy view and give its space to the EDT view
    robot_viz.sensor_circle.set_visible(False)
    robot_viz.ax_local.remove()
    robot_viz.ax_edt.set_subplotspec(robot_viz.ax_edt.get_subplotspec().get_gridspec()[0, 1:])

    dt = 0.1

    path = [(robot_pose[0], robot_pose[1])]

    with udp:
        while True:
            free_mask = global_free_mask(obstacle_grid, robot_pose, sensor_radius)

            # only the global goal, which is the final one, and no route nodes
            u, field, kernels, kernel_gamma, controller_edt, _ = robot_controller_step(udp, robot_pose, goal, sensor_radius, free_mask, [], robot_viz, is_final=True)

            robot_viz.update_vector_field(field[:, :2], field[:, 2:])
            robot_viz.update_kernels(kernels[:, :2], kernels[:, 2:], gamma=kernel_gamma)

            robot_pose = dd_step(robot_pose, u, dt)
            path.append((robot_pose[0], robot_pose[1]))

            if Utils.check_collision(obstacle_grid, robot_pose, robot_radius):
                print(f"[Py] collision with an obstacle at ({robot_pose[0]:.1f}, {robot_pose[1]:.1f}) after {len(path) - 1} ticks, stopping", flush=True)
                udp.controller_stop()
                return

            robot_viz.update_trajectory(path)
            robot_viz.update(robot_pose=robot_pose, pause=0.1)
            show_controller_edt(robot_viz, controller_edt, obstacle_grid)

            if Utils.calc_dist((robot_pose[0], robot_pose[1]), goal) < robot_radius:
                print(f"[Py] goal reached after {len(path) - 1} ticks", flush=True)
                udp.controller_stop()

DISPATCH = {
    "robot": run_robot,
    "car": run_car,
    "social_cross": run_social_cross,
    "social_counter": run_social_counter,
    "global": run_global,
}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run a graph-guided reactive navigation mode.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default=DEFAULT_MODE,
        choices=MODES,
        help=f"navigation mode to run (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "-m",
        "--map",
        type=int,
        default=None,
        choices=MAP_NUMBERS,
        metavar="N",
        help=(
            "map to load, one of "
            f"{{{', '.join(str(n) for n in MAP_NUMBERS)}}} "
            f"(default: {DEFAULT_MAP}); "
            f"only valid for mode(s): {', '.join(MAP_MODES)}"
        ),
    )

    args = parser.parse_args(argv)

    if args.map is not None and args.mode not in MAP_MODES:
        parser.error(
            f"--map is not supported for mode '{args.mode}'; "
            f"it only applies to: {', '.join(MAP_MODES)}"
        )

    if args.map is None and args.mode in MAP_MODES:
        args.map = DEFAULT_MAP

    return args


def main(argv=None):
    args = parse_args(argv)

    if args.mode in MAP_MODES:
        DISPATCH[args.mode](map_number=args.map)
    else:
        DISPATCH[args.mode]()

    return 0


if __name__ == "__main__":
    sys.exit(main())
