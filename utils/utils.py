import numpy as np
from matplotlib.path import Path

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

        return grid

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

        return grid
   
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

        return grid
    
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

        return grid
    
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

        return grid

    @staticmethod
    def get_map6(shape=(120, 60)):
        grid = np.zeros(shape, dtype=bool)
        
        columns, rows = shape

        border = 1
        
        MapCreator.__add_rect(grid, 0, border, 0, rows)
        MapCreator.__add_rect(grid, columns - border, columns, 0, rows)
        MapCreator.__add_rect(grid, 0, columns, 0, border)
        MapCreator.__add_rect(grid, 0, columns, rows - border, rows)

        # ---------------------------------------------------------
        # Main vertical maze walls
        # ---------------------------------------------------------

        # Wall 1 - opening at bottom
        MapCreator.__add_rect(grid, 22, 26, 13, 53)

        # Wall 2 - opening at top
        MapCreator.__add_rect(grid, 34, 38, 7, 47)

        # Wall 3 - opening at bottom
        MapCreator.__add_rect(grid, 46, 50, 13, 53)

        # Wall 4 - opening at top
        MapCreator.__add_rect(grid, 58, 62, 7, 47)

        # Wall 5 - opening at bottom
        MapCreator.__add_rect(grid, 70, 74, 13, 53)

        # Wall 6 - opening at top
        MapCreator.__add_rect(grid, 82, 86, 7, 47)

        # Wall 7 - opening at bottom
        MapCreator.__add_rect(grid, 94, 98, 13, 53)

        # ---------------------------------------------------------
        # Dead-end branches
        # ---------------------------------------------------------

        # Branch 1
        MapCreator.__add_rect(grid, 14, 22, 23, 27)

        # Branch 2
        MapCreator.__add_rect(grid, 26, 34, 35, 39)

        # Branch 3
        MapCreator.__add_rect(grid, 38, 46, 19, 23)

        # Branch 4
        MapCreator.__add_rect(grid, 50, 58, 35, 39)

        # Branch 5
        MapCreator.__add_rect(grid, 62, 70, 19, 23)

        # Branch 6
        MapCreator.__add_rect(grid, 74, 82, 35, 39)

        # Branch 7
        MapCreator.__add_rect(grid, 86, 94, 19, 23)

        # Branch 8
        MapCreator.__add_rect(grid, 98, 106, 35, 39)

        return grid

    
    @staticmethod
    def get_map7(shape=(100, 50)):

        grid = np.zeros(shape, dtype=bool)

        columns, rows = shape

        # ---------------------------------------------------------
        # Outer boundary
        # ---------------------------------------------------------
        border = 1

        MapCreator.__add_rect(grid, 0, border, 0, rows)
        MapCreator.__add_rect(grid, columns - border, columns, 0, rows)
        MapCreator.__add_rect(grid, 0, columns, 0, border)
        MapCreator.__add_rect(grid, 0, columns, rows - border, rows)

        # ---------------------------------------------------------
        # Main alternating maze walls
        #
        # Each wall almost spans the complete height.
        # The opening alternates between bottom and top.
        # ---------------------------------------------------------

        wall_width = 3

        # Bottom opening
        MapCreator.__add_rect(grid, 10, 10 + wall_width, 10, 49)

        # Top opening
        MapCreator.__add_rect(grid, 22, 22 + wall_width, 1, 40)

        # Bottom opening
        MapCreator.__add_rect(grid, 34, 34 + wall_width, 10, 49)

        # Top opening
        MapCreator.__add_rect(grid, 46, 46 + wall_width, 1, 40)

        # Bottom opening
        MapCreator.__add_rect(grid, 58, 58 + wall_width, 10, 49)

        # Top opening
        MapCreator.__add_rect(grid, 70, 70 + wall_width, 1, 40)

        # Bottom opening
        MapCreator.__add_rect(grid, 82, 82 + wall_width, 10, 49)

        # ---------------------------------------------------------
        # Dead-end side branches
        # ---------------------------------------------------------

        # Upper dead ends
        MapCreator.__add_rect(grid, 5, 10, 17, 20)
        MapCreator.__add_rect(grid, 29, 34, 27, 30)
        MapCreator.__add_rect(grid, 53, 58, 17, 20)
        MapCreator.__add_rect(grid, 77, 82, 27, 30)

        # Lower dead ends
        MapCreator.__add_rect(grid, 17, 22, 31, 34)
        MapCreator.__add_rect(grid, 41, 46, 17, 20)
        MapCreator.__add_rect(grid, 65, 70, 31, 34)
        MapCreator.__add_rect(grid, 89, 95, 17, 20)

        return grid