import math
import numpy as np
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt

class Utils:
    @staticmethod
    def calc_dist(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    @staticmethod
    def check_collision(obstacle_grid, robot_pose, robot_radius):
        # True when the robot disc overlaps an occupied cell, each cell is the unit square around its index (as drawn on the map)
        # cells outside the map count as occupied
        columns, rows = obstacle_grid.shape
        x, y = robot_pose[0], robot_pose[1]

        # cells whose square the disc can reach
        xx, yy = np.ogrid[int(np.ceil(x - robot_radius - 0.5)):int(np.floor(x + robot_radius + 0.5)) + 1,
                          int(np.ceil(y - robot_radius - 0.5)):int(np.floor(y + robot_radius + 0.5)) + 1]

        # distance from the robot centre to the nearest point of each cell square
        dx = np.maximum(np.abs(xx - x) - 0.5, 0.0)
        dy = np.maximum(np.abs(yy - y) - 0.5, 0.0)
        touching = dx ** 2 + dy ** 2 < robot_radius ** 2

        in_map = (xx >= 0) & (xx < columns) & (yy >= 0) & (yy < rows)
        occupied = ~in_map | obstacle_grid[np.clip(xx, 0, columns - 1), np.clip(yy, 0, rows - 1)]

        return bool(np.any(touching & occupied))

    @staticmethod
    def check_car_collision(obstacle_grid, robot_pose, footprint):
        # True when the car's box overlaps an occupied cell, each cell is the unit square around its index (as drawn on the map)
        # cells outside the map count as occupied
        # footprint: (front, rear, half width) from the car origin along its heading; the box and a cell square overlap when
        # no separating axis exists among the map axes and the box's two axes
        columns, rows = obstacle_grid.shape
        x, y, theta = robot_pose[0], robot_pose[1], robot_pose[2]
        front, rear, half_width = footprint

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        half_length = 0.5 * (front + rear)

        # box centre, the origin is not in its middle when front != rear
        center_x = x + 0.5 * (front - rear) * cos_t
        center_y = y + 0.5 * (front - rear) * sin_t

        # half extents of the box along the map axes, and of a cell square along the box axes
        extent_x = half_length * abs(cos_t) + half_width * abs(sin_t)
        extent_y = half_length * abs(sin_t) + half_width * abs(cos_t)
        square_extent = 0.5 * (abs(cos_t) + abs(sin_t))

        # cells whose square the box's bounding box can reach
        xx, yy = np.ogrid[int(np.ceil(center_x - extent_x - 0.5)):int(np.floor(center_x + extent_x + 0.5)) + 1,
                          int(np.ceil(center_y - extent_y - 0.5)):int(np.floor(center_y + extent_y + 0.5)) + 1]

        dx = xx - center_x
        dy = yy - center_y
        touching = ((np.abs(dx) < extent_x + 0.5) & (np.abs(dy) < extent_y + 0.5)
                    & (np.abs(dx * cos_t + dy * sin_t) < half_length + square_extent)
                    & (np.abs(-dx * sin_t + dy * cos_t) < half_width + square_extent))

        in_map = (xx >= 0) & (xx < columns) & (yy >= 0) & (yy < rows)
        occupied = ~in_map | obstacle_grid[np.clip(xx, 0, columns - 1), np.clip(yy, 0, rows - 1)]

        return bool(np.any(touching & occupied))


class PolygonOps:
    @staticmethod
    def get_polygon_centroid(polygon: np.ndarray) -> tuple[float, float]:
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = PolygonOps.get_polygon_area(polygon)

        cross = x * np.roll(y, -1) - np.roll(x, -1) * y

        centroid_x = float(np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area))
        centroid_y = float(np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area))

        return centroid_x, centroid_y
    
    @staticmethod
    def get_polygon_area(polygon: np.ndarray) -> float:
        x = polygon[:, 0]
        y = polygon[:, 1]

        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    
    @staticmethod
    def ensure_ccw(polygon: np.ndarray) -> np.ndarray:
        x = polygon[:, 0]
        y = polygon[:, 1]

        signed_area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

        return polygon if signed_area > 0 else polygon[::-1]

    @staticmethod
    def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

class MapCreator:
    @staticmethod
    def __add_rect(grid, x1, x2, y1, y2):
        grid[x1:x2, y1:y2] = True

    @staticmethod
    def __add_circle(grid, center_x, center_y, radius):
        columns, rows = grid.shape
        xx, yy = np.ogrid[:columns, :rows]
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        grid[mask] = True
    
    @staticmethod
    def __add_polygon(grid: np.ndarray, vertices: list[tuple[float, float]]):

        if grid.ndim != 2:
            raise ValueError("grid must be a two-dimensional array.")

        if len(vertices) < 3:
            raise ValueError("A polygon requires at least three vertices.")

        columns, rows = grid.shape

        x_coordinates, y_coordinates = np.mgrid[0:columns, 0:rows]

        cell_centers = np.column_stack((x_coordinates.ravel(), y_coordinates.ravel()))

        polygon = Path(vertices)

        polygon_mask = polygon.contains_points(cell_centers, radius=0.5).reshape(columns, rows)

        grid[polygon_mask] = True

    @staticmethod
    def __add_thick_segment(grid: np.ndarray, start: tuple[float, float], end: tuple[float, float], thickness: float):

        if thickness <= 0:
            raise ValueError("thickness must be positive.")

        columns, rows = grid.shape

        x_coordinates, y_coordinates = np.mgrid[0:columns, 0:rows]

        x_start, y_start = start
        x_end, y_end = end

        delta_x = x_end - x_start
        delta_y = y_end - y_start

        segment_length_squared = (delta_x**2 + delta_y**2)

        if segment_length_squared == 0:
            distance_squared = ((x_coordinates - x_start) ** 2 + (y_coordinates - y_start) ** 2)

        else:
            projection = ((x_coordinates - x_start) * delta_x + (y_coordinates - y_start) * delta_y) / segment_length_squared

            # Restrict the projection to the finite line segment.
            projection = np.clip(projection, 0.0, 1.0)

            closest_x = (x_start + projection * delta_x)

            closest_y = (y_start + projection * delta_y)

            distance_squared = ((x_coordinates - closest_x) ** 2 + (y_coordinates - closest_y) ** 2)

        radius = thickness / 2.0

        grid[distance_squared <= radius**2] = True

    @staticmethod
    def __add_thick_polyline(grid: np.ndarray, points: list[tuple[float, float]], thickness: float) -> None:

        if len(points) < 2:
            raise ValueError(
                "A polyline requires at least two points."
            )

        for start, end in zip(points[:-1], points[1:]):
            MapCreator.__add_thick_segment(grid=grid, start=start, end=end, thickness=thickness)

    @staticmethod
    def get_map1(shape=(120, 80)):
        """
        Map with 2 large rooms and a narrow middle corridor.
        """
        grid = np.zeros(shape, dtype=bool)

        # top and bottom walls to shape corridor-style free space
        MapCreator.__add_rect(grid, 35, 85, 0, 20)
        MapCreator.__add_rect(grid, 35, 85, 60, 80)

        # central vertical obstacle leaving a narrow passage
        MapCreator.__add_rect(grid, 54, 66, 20, 60)

        # small internal obstacles in left and right rooms
        MapCreator.__add_circle(grid, 20, 28, 8)
        MapCreator.__add_circle(grid, 22, 52, 7)
        MapCreator.__add_circle(grid, 100, 28, 8)
        MapCreator.__add_circle(grid, 98, 52, 7)

        start = (20, 10)
        goal = (20, 70)
        
        return grid, start, goal

    @staticmethod
    def get_map2(shape=(120, 80)):
        """
        Maze-like map with rectangular barriers.
        """
        grid = np.zeros(shape, dtype=bool)

        MapCreator.__add_rect(grid, 20, 28, 10, 70)
        MapCreator.__add_rect(grid, 45, 53, 0, 50)
        MapCreator.__add_rect(grid, 70, 78, 30, 80)
        MapCreator.__add_rect(grid, 95, 103, 10, 55)

        MapCreator.__add_rect(grid, 0, 55, 0, 12)
        MapCreator.__add_rect(grid, 30, 120, 68, 80)

        start = (110, 30)
        goal = (40, 25)
                
        return grid, start, goal
   
    @staticmethod
    def get_map3(shape=(120, 80)):
        """
        Mixed circular and rectangular obstacles.
        """
        grid = np.zeros(shape, dtype=bool)

        # circular obstacles
        MapCreator.__add_circle(grid, 20, 18, 10)
        MapCreator.__add_circle(grid, 55, 20, 9)
        MapCreator.__add_circle(grid, 95, 18, 10)

        MapCreator.__add_circle(grid, 25, 58, 11)
        MapCreator.__add_circle(grid, 60, 58, 9)
        MapCreator.__add_circle(grid, 98, 55, 11)

        # rectangles
        MapCreator.__add_rect(grid, 0, 35, 30, 40)
        MapCreator.__add_rect(grid, 45, 75, 35, 45)
        MapCreator.__add_rect(grid, 85, 120, 30, 40)

        start = (112, 75)
        goal = (30, 5)
                        
        return grid, start, goal
    
    @staticmethod
    def get_map4(shape=(180, 100)):
        """
        Map with two outward-facing C-shaped obstacles and two central tapered barriers.
        """
        grid = np.zeros(shape, dtype=bool)

        # Map border.
        border = 2
        grid[:border, :] = True
        grid[-border:, :] = True
        grid[:, :border] = True
        grid[:, -border:] = True

        # Left C-shaped obstacle.
        # The opening faces the left side of the map.
        left_wall = [
            (18, 64),
            (26, 76),
            (42, 84),
            (56, 78),
            (65, 66),
            (65, 34),
            (56, 22),
            (42, 16),
            (26, 24),
            (18, 36),
        ]

        # Right C-shaped obstacle.
        # The opening faces the right side of the map.
        right_wall = [
            (162, 64),
            (154, 76),
            (138, 84),
            (124, 78),
            (115, 66),
            (115, 34),
            (124, 22),
            (138, 16),
            (154, 24),
            (162, 36),
        ]

        wall_thickness = 6.0

        MapCreator.__add_thick_polyline(grid, left_wall, wall_thickness)

        MapCreator.__add_thick_polyline(grid, right_wall, wall_thickness)

        # Upper central tapered obstacle.
        upper_barrier = [
            (74, 99),
            (106, 99),
            (101, 87),
            (97, 61),
            (83, 61),
            (79, 87),
        ]

        MapCreator.__add_polygon(grid, upper_barrier)

        # Lower central tapered obstacle.
        lower_barrier = [
            (74, 0),
            (106, 0),
            (101, 13),
            (97, 39),
            (83, 39),
            (79, 13),
        ]

        MapCreator.__add_polygon(grid, lower_barrier)

        start = (40, 50)
        goal = (140, 50)
        
        return grid, start, goal
    
    @staticmethod
    def get_map5(shape=(180, 100)):
        grid = np.zeros(shape, dtype=bool)

        columns, rows = shape

        # Map border.
        border = 2

        MapCreator.__add_rect(grid, 0, border, 0, rows)

        MapCreator.__add_rect(grid, columns - border, columns, 0, rows)

        MapCreator.__add_rect(grid, 0, columns, 0, border)

        MapCreator.__add_rect(grid, 0, columns, rows - border, rows)

        left_x_min = 20
        left_x_max = 65

        lower_y_min = 20
        lower_y_max = 30

        upper_y_min = 70
        upper_y_max = 80

        back_wall_width = 10

        # Lower horizontal arm.
        MapCreator.__add_rect(grid, left_x_min, left_x_max, lower_y_min, lower_y_max)

        # Upper horizontal arm.
        MapCreator.__add_rect(grid, left_x_min, left_x_max, upper_y_min, upper_y_max)

        # Right vertical back wall.
        MapCreator.__add_rect(grid, left_x_max - back_wall_width, left_x_max, lower_y_min, upper_y_max)

        right_x_min = 115
        right_x_max = 160

        # Lower horizontal arm.
        MapCreator.__add_rect(grid, right_x_min, right_x_max, lower_y_min, lower_y_max)

        # Upper horizontal arm.
        MapCreator.__add_rect(grid, right_x_min, right_x_max, upper_y_min, upper_y_max)

        # Left vertical back wall.
        MapCreator.__add_rect(grid, right_x_min, right_x_min + back_wall_width, lower_y_min, upper_y_max)

        central_x_min = 80
        central_x_max = 100

        # Upper central barrier.
        MapCreator.__add_rect(grid, central_x_min, central_x_max, 60, rows)

        # Lower central barrier.
        MapCreator.__add_rect(grid, central_x_min, central_x_max, 0, 40)

        start = (40, 50)
        goal = (140, 50)

        return grid, start, goal

    @staticmethod
    def get_map6(shape=(182, 104)):
        grid = np.zeros(shape, dtype=bool)

        columns, rows = shape

        cell = 26
        half_wall = 2

        MapCreator.__add_rect(grid, 0, half_wall, 0, rows)
        MapCreator.__add_rect(grid, columns - half_wall, columns, 0, rows)
        MapCreator.__add_rect(grid, 0, columns, 0, half_wall)
        MapCreator.__add_rect(grid, 0, columns, rows - half_wall, rows)

        # Walls between maze cells, in maze cell units:
        # (x boundary, first row, last row).
        vertical_walls = [
            (1, 1, 1),
            (2, 1, 2),
            (3, 2, 2),
            (4, 2, 3),
            (5, 0, 2),
            (6, 1, 3),
        ]

        # (y boundary, first column, last column).
        horizontal_walls = [
            (1, 1, 4),
            (2, 3, 3),
            (3, 1, 1),
        ]

        # Each wall also covers the posts at both of its ends. Clamp at 0 so a
        # wall starting at the border does not become a negative (wrapping) index.
        for x_boundary, first_row, last_row in vertical_walls:
            MapCreator.__add_rect(
                grid,
                x_boundary * cell - half_wall,
                x_boundary * cell + half_wall,
                max(first_row * cell - half_wall, 0),
                (last_row + 1) * cell + half_wall,
            )

        for y_boundary, first_column, last_column in horizontal_walls:
            MapCreator.__add_rect(
                grid,
                max(first_column * cell - half_wall, 0),
                (last_column + 1) * cell + half_wall,
                y_boundary * cell - half_wall,
                y_boundary * cell + half_wall,
            )

        start = (13, 13)
        goal = (170, 90)

        return grid, start, goal

    @staticmethod
    def get_counterflow_map(shape=(120, 80), corridor_rows=(24, 56)):
        # Horizontal corridor with a wall above and below it, for the counterflow
        # pedestrian bottleneck: the robot drives along it while pedestrians walk
        # the other way, so the only way past them is around them.
        grid = np.zeros(shape, dtype=bool)

        columns, rows = shape

        first_free_row, last_free_row = corridor_rows

        if not 0 <= first_free_row <= last_free_row < rows:
            raise ValueError("corridor_rows must be two rows of the map, the lower one first.")

        # Everything outside the free rows is wall; the map border closes the corridor's ends.
        MapCreator.__add_rect(grid, 0, columns, 0, first_free_row)

        MapCreator.__add_rect(grid, 0, columns, last_free_row + 1, rows)

        return grid

    @staticmethod
    def stamp_moving_obstacle(static_grid, center_x, center_y, half_size=2):
        size = 2 * half_size + 1

        x0 = int(round(center_x)) - half_size
        y0 = int(round(center_y)) - half_size
        x0 = max(0, min(static_grid.shape[0] - size, x0))
        y0 = max(0, min(static_grid.shape[1] - size, y0))

        block_mask = np.zeros_like(static_grid)
        block_mask[x0:x0 + size, y0:y0 + size] = True

        return static_grid | block_mask, block_mask, x0, y0

    @staticmethod
    def get_u_trap_map(width=100, height=60):
        # smaller copy of the left U of MapCreator.get_map5, centred in the map: open to the left, the robot starts in front
        # of it and the goal is behind the back wall
        # returns the grid, the start and the goal
        grid = np.zeros((width, height), dtype=bool)

        # Map border.
        border = 2

        MapCreator.__add_rect(grid, 0, border, 0, height)

        MapCreator.__add_rect(grid, width - border, width, 0, height)

        MapCreator.__add_rect(grid, 0, width, 0, border)

        MapCreator.__add_rect(grid, 0, width, height - border, height)

        # U of the 64 x 64 map (32 cells long, 34 tall, 4-cell walls), centred in the map
        u_length = 32
        u_height = 34
        arm_width = 4

        x_min = (width - u_length) // 2
        x_max = x_min + u_length

        lower_y_min = (height - u_height) // 2
        lower_y_max = lower_y_min + arm_width

        upper_y_max = lower_y_min + u_height
        upper_y_min = upper_y_max - arm_width

        back_wall_width = 4

        # Lower horizontal arm.
        MapCreator.__add_rect(grid, x_min, x_max, lower_y_min, lower_y_max)

        # Upper horizontal arm.
        MapCreator.__add_rect(grid, x_min, x_max, upper_y_min, upper_y_max)

        # Right vertical back wall.
        MapCreator.__add_rect(grid, x_max - back_wall_width, x_max, lower_y_min, upper_y_max)

        # start in front of the opening, goal halfway between the back wall and the right border,
        # both 2 cells below the U's axis as in the 64 x 64 map
        y = (lower_y_min + upper_y_max) // 2 - 2
        start = (5, y)
        goal = ((x_max + width - border) // 2, y)

        return grid, start, goal



class MovingObstacles:
    """Disc-shaped moving obstacles (pedestrians) walking in straight lines between random free points of a static map."""

    def __init__(self, static_grid: np.ndarray, count: int = 10, radius: float = 2.0, speed_range: tuple[float, float] = (1.5, 3.5), keep_out: list[tuple[float, float]] | None = None, keep_out_radius: float = 10.0, yield_to_robot: bool = False, seed: int = 1) -> None:
        self.static_grid = static_grid
        self.radius = radius
        self.yield_to_robot = yield_to_robot
        self.rng = np.random.default_rng(seed)

        # Cells where a pedestrian fits without touching a static obstacle.
        self.clearance = distance_transform_edt(~static_grid)
        self.free_cells = np.argwhere(self.clearance > radius + 1.0).astype(float)

        if len(self.free_cells) == 0:
            raise ValueError("The map has no free space for the moving obstacles.")

        keep_out = keep_out or []
        starts: list[np.ndarray] = []

        # Start away from the given points (robot start and goal) and from each other.
        for _ in range(100000):
            if len(starts) == count:
                break

            point = self._random_free_point()

            far_from_keep_out = all(np.hypot(point[0] - x, point[1] - y) > keep_out_radius for x, y in keep_out)
            far_from_others = all(np.hypot(*(point - other)) > 2.0 * radius + 1.0 for other in starts)

            if far_from_keep_out and far_from_others:
                starts.append(point)
        else:
            raise ValueError("Could not place all moving obstacles, use fewer of them or a smaller keep-out radius.")

        self.positions = np.array(starts).reshape(-1, 2)
        self.goals = np.array([self._random_free_point() for _ in range(count)]).reshape(-1, 2)
        self.speeds = self.rng.uniform(speed_range[0], speed_range[1], count)
        self.velocities = np.zeros((count, 2))

    def _random_free_point(self) -> np.ndarray:
        return self.free_cells[self.rng.integers(len(self.free_cells))].copy()

    def _fits(self, point: np.ndarray) -> bool:
        columns, rows = self.static_grid.shape

        x = int(round(point[0]))
        y = int(round(point[1]))

        if not (0 <= x < columns and 0 <= y < rows):
            return False

        return self.clearance[x, y] > self.radius + 0.5

    def step(self, dt: float, robot_pose: tuple[float, float, float] | None = None, robot_radius: float = 0.0) -> None:
        for k in range(len(self.positions)):
            to_goal = self.goals[k] - self.positions[k]
            distance = np.hypot(*to_goal)
            step_length = self.speeds[k] * dt

            # Goal reached: walk to a new random point.
            if distance <= step_length:
                self.positions[k] = self.goals[k]
                self.goals[k] = self._random_free_point()
                self.velocities[k] = 0.0
                continue

            next_position = self.positions[k] + to_goal / distance * step_length

            blocked = not self._fits(next_position)

            for j in range(len(self.positions)):
                if j != k and np.hypot(*(next_position - self.positions[j])) < 2.0 * self.radius:
                    blocked = True

            # A static obstacle or another pedestrian is in the way: walk somewhere else.
            if blocked:
                self.goals[k] = self._random_free_point()
                self.velocities[k] = 0.0
                continue

            # A polite pedestrian waits instead of walking into the robot.
            if self.yield_to_robot and robot_pose is not None:
                if np.hypot(next_position[0] - robot_pose[0], next_position[1] - robot_pose[1]) < self.radius + robot_radius + 0.5:
                    self.velocities[k] = 0.0
                    continue

            self.velocities[k] = (next_position - self.positions[k]) / dt
            self.positions[k] = next_position

    def grid(self) -> np.ndarray:
        # Static map with every pedestrian stamped as a filled disc.
        grid = self.static_grid.copy()

        columns, rows = grid.shape
        cells = int(np.ceil(self.radius))

        for center_x, center_y in self.positions:
            x = int(round(center_x))
            y = int(round(center_y))

            x_start, x_end = max(0, x - cells), min(columns, x + cells + 1)
            y_start, y_end = max(0, y - cells), min(rows, y + cells + 1)

            xx, yy = np.ogrid[x_start:x_end, y_start:y_end]
            grid[x_start:x_end, y_start:y_end] |= (xx - center_x) ** 2 + (yy - center_y) ** 2 <= self.radius ** 2

        return grid

    def min_distance(self, robot_pose: tuple[float, float, float]) -> float:
        # Distance from the robot centre to the nearest pedestrian centre.
        return float(np.min(np.hypot(self.positions[:, 0] - robot_pose[0], self.positions[:, 1] - robot_pose[1])))


class CounterflowPedestrians(MovingObstacles):

    # Coordinate the pedestrians walk along: 0 = x (horizontal lanes), 1 = y (vertical lanes, CrossflowPedestrians).
    axis = 0

    def __init__(self, static_grid: np.ndarray, positions, velocities, radius: float = 2.0, x_bounds=None) -> None:
        self._setup(static_grid, positions, velocities, radius, x_bounds)

    def _setup(self, static_grid: np.ndarray, positions, velocities, radius: float, bounds) -> None:
        # bounds: where each pedestrian turns around on the coordinate it walks along
        self.static_grid = static_grid
        self.radius = radius
        # The pedestrians never react to the robot, so the walk is the same for every controller.
        self.yield_to_robot = False

        # Clearance to the static obstacles, used by _fits() to keep the pedestrians in the corridor.
        self.clearance = distance_transform_edt(~static_grid)

        self.positions = np.array(positions, dtype=float).reshape(-1, 2)
        self.velocities = np.array(velocities, dtype=float).reshape(-1, 2)

        count = len(self.positions)

        if len(self.velocities) != count:
            raise ValueError("Every pedestrian needs one velocity.")

        # The velocity along the walking axis: its size is the walking speed, its sign the direction the pedestrian walks in.
        self.speeds = np.abs(self.velocities[:, self.axis])
        self.directions = np.where(self.velocities[:, self.axis] < 0.0, -1.0, 1.0)

        # The lane each pedestrian walks on (the other coordinate); it never leaves it.
        self.lanes = self.positions[:, 1 - self.axis].copy()

        # By default a pedestrian turns around only where it would leave the map.
        if bounds is None:
            bounds = [(radius, static_grid.shape[self.axis] - 1.0 - radius)] * count

        self.bounds = np.array(bounds, dtype=float).reshape(-1, 2)

        if len(self.bounds) != count:
            raise ValueError(f"Every pedestrian needs one pair of {'xy'[self.axis]} bounds.")

        # Simulation time walked so far.
        self.time = 0.0

        self._check_starts()

    def _check_starts(self) -> None:
        name = "xy"[self.axis]

        for k, (x, y) in enumerate(self.positions):
            lower, upper = self.bounds[k]
            along = self.positions[k, self.axis]

            if lower > upper:
                raise ValueError(f"Pedestrian {k} has {name} bounds [{lower:g}, {upper:g}], the lower one first.")

            if not lower <= along <= upper:
                raise ValueError(f"Pedestrian {k} starts at {name} = {along:g}, outside its {name} bounds [{lower:g}, {upper:g}].")

            if not self._fits(self.positions[k]):
                raise ValueError(f"Pedestrian {k} starts at ({x:g}, {y:g}), which is not free space for its radius.")

            for j in range(k):
                if np.hypot(*(self.positions[k] - self.positions[j])) < 2.0 * self.radius:
                    raise ValueError(f"Pedestrians {j} and {k} overlap at their starting positions.")

    def step(self, dt: float, robot_pose: tuple[float, float, float] | None = None, robot_radius: float = 0.0) -> None:
        # robot_pose and robot_radius are taken to keep the MovingObstacles interface, but are never read:
        # these pedestrians do not react to the robot.
        self.time += dt

        for k in range(len(self.positions)):
            # position along the walking axis
            along = float(self.positions[k, self.axis])
            direction = float(self.directions[k])
            remaining = float(self.speeds[k]) * dt

            lower, upper = self.bounds[k]

            # Walk the step in pieces, turning around at a bound rather than stepping over it,
            # so a long step reverses on the spot instead of jumping outside the bounds.
            while remaining > 1e-12:
                bound = upper if direction > 0.0 else lower
                to_bound = abs(bound - along)

                if remaining < to_bound:
                    along += direction * remaining
                    remaining = 0.0
                else:
                    along = bound
                    remaining -= to_bound
                    direction = -direction

            next_position = np.empty(2)
            next_position[self.axis] = along
            next_position[1 - self.axis] = self.lanes[k]

            others = np.delete(self.positions, k, axis=0)

            overlapping = len(others) > 0 and bool(np.any(np.hypot(*(others - next_position).T) < 2.0 * self.radius))

            # Separate lanes and bounds normally keep the pedestrians apart; if a step would still
            # put one on top of another or into a wall, it turns around where it stands.
            if overlapping or not self._fits(next_position):
                direction = -direction
            else:
                self.positions[k] = next_position

            self.directions[k] = direction
            self.velocities[k] = 0.0
            self.velocities[k, self.axis] = direction * self.speeds[k]


class CrossflowPedestrians(CounterflowPedestrians):

    axis = 1

    def __init__(self, static_grid: np.ndarray, positions, velocities, radius: float = 2.0, y_bounds=None) -> None:
        self._setup(static_grid, positions, velocities, radius, y_bounds)
