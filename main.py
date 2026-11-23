import argparse
import sys
from utils.utils import MapCreator
from global_planner.graph_generator import RectNodeGraphGenerator
from global_planner.route_planner import DijkstraRoutePlanner
from model.robot_config import RobotConfig, CarConfig
from utils.ui import Visualizer

DEFAULT_MODE = "robot"
MODES = ("robot", "car", "social_cross", "social_counter", "global")

MAP_MODES = ("robot")

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

def get_global_route(map_number=DEFAULT_MAP, min_node_half_size=RobotConfig.robot_radius):
    obstacle_grid, start, goal = get_map(map_number)

    dst_thr = 2.0

    rectNodegenerator = RectNodeGraphGenerator(obstacle_grid=obstacle_grid, alpha=0.99, p_confidence=0.90, gamma=1.05, dst_thr=dst_thr, robot_radius=min_node_half_size)

    result = rectNodegenerator.generate()

    nodes = result['nodes']
    graph = result['graph']

    route_planner = DijkstraRoutePlanner(nodes=nodes, graph=graph)
    route_planner.find_route(start=start, goal=goal)

    waypoints = route_planner.get_pruned_temp_goal_points()
    route_nodes = [nodes[node_id] for node_id in route_planner.pruned_route_node_ids]

    return waypoints, route_nodes, obstacle_grid, start, goal

def get_map(map_number):
    return getattr(MapCreator, f"get_map{map_number}")()


def run_robot(map_number=DEFAULT_MAP):
    print(f"[mode] robot (map {map_number})")
    min_node_half_size = RobotConfig.robot_radius
    waypoints, route_nodes, obstacle_grid, start, goal = get_global_route(map_number=map_number, min_node_half_size=min_node_half_size)
    robot_pose = [start[0], start[1], 0.0]
    robot_radius = RobotConfig.robot_radius
    sensor_radius = RobotConfig.sensor_radius
    fig, ax, robot_viz = Visualizer.visualize_graph(obstacle_grid=obstacle_grid, nodes=route_nodes, temp_goals=waypoints, goal=goal, start=start, robot_pose=robot_pose, robot_radius=robot_radius, sensor_radius=sensor_radius, block=False, save_global_map=False, save_local_map=False, save_edt_map=False, footprint=None)
    while True:
        robot_viz.update(robot_pose=robot_pose, pause=0.1)

def run_car():
    footprint = (CarConfig.car_front, CarConfig.car_rear, CarConfig.car_half_width)
    # robot_pose = [start[0], start[1], 0.0, 0.0]
    print("[mode] car")

def run_social_cross():
    print("[mode] social_cross")

def run_social_counter():
    print("[mode] social_counter")

def run_global():
    print("[mode] global")

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
