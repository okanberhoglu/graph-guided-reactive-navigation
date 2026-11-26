from dataclasses import dataclass
import numpy as np
import math

# Gridcell position -> (x, y)
GridCell = tuple[int, int]

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

@dataclass
class RectRegion:

    center_x: float
    center_y: float

    theta: float

    half_width: float
    half_height: float

    @property
    def width(self) -> float:
        return 2.0 * self.half_width
    
    @property
    def height(self) -> float:
        return 2.0 * self.half_height
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> np.ndarray:
         return np.array([self.center_x, self.center_y], dtype=float)
    
    @property
    def axis_1(self) -> np.ndarray:
        # Unit vector along the rectangle's width direction.
        return np.array([math.cos(self.theta), math.sin(self.theta)], dtype=float)
    
    @property
    def axis_2(self) -> np.ndarray:
        # Unit vector perpendicular to axis_1.
        return np.array([-math.sin(self.theta), math.cos(self.theta)], dtype=float)
    
    @property
    def vertices(self) -> np.ndarray:
        width_vector = self.half_width * self.axis_1
        height_vector = self.half_height * self.axis_2
        center = self.center

        return np.array(
            [
                center + width_vector + height_vector,
                center - width_vector + height_vector,
                center - width_vector - height_vector,
                center + width_vector - height_vector,
            ], dtype=float)

    def contains(self, cell: GridCell) -> bool:
        x, y = cell

        point = np.array([float(x), float(y)])

        displacement = point - self.center

        local_coordinate_1 = float(displacement @ self.axis_1)

        local_coordinate_2 = float(displacement @ self.axis_2)

        return (abs(local_coordinate_1) <= self.half_width and abs(local_coordinate_2) <= self.half_height)
    
    def intersects(self, other_node: "RectRegion", tolerance: float = 1e-9) -> bool:
        """
        Check whether two oriented rectangles overlap.

        Uses the Separating Axis Theorem.

        Merely touching at an edge or corner is not considered an
        intersection because MPC-Graph requires an overlap region.
        """

        self_vertices = self.vertices
        other_vertices = other_node.vertices

        test_axes = [self.axis_1, self.axis_2, other_node.axis_1, other_node.axis_2]

        for axis in test_axes:
            self_projection = self_vertices @ axis
            other_projection = other_vertices @ axis

            self_min = float(self_projection.min())
            self_max = float(self_projection.max())

            other_min = float(other_projection.min())
            other_max = float(other_projection.max())

            if self_max <= other_min + tolerance:
                return False

            if other_max <= self_min + tolerance:
                return False
            
        return True
    
    def get_intersection_polygon(self, node: "RectRegion") -> "np.ndarray | None":
        """
        Clip this rectangle against `node` (Sutherland-Hodgman) and return
        the overlap polygon's vertices as an (N, 2) array of (x, y) points,
        or None if the rectangles do not overlap in a proper polygon.
        """
        output = list(PolygonOps.ensure_ccw(self.vertices))
        clip_polygon = PolygonOps.ensure_ccw(node.vertices)

        for i in range(len(clip_polygon)):
            if not output:
                return None

            edge_start = clip_polygon[i]
            edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
            edge_direction = edge_end - edge_start

            input_list = output
            output = []

            for j in range(len(input_list)):
                current = input_list[j]
                previous = input_list[j - 1]

                current_inside = PolygonOps.cross_2d(edge_direction, current - edge_start) >= 0
                previous_inside = PolygonOps.cross_2d(edge_direction, previous - edge_start) >= 0

                if current_inside:
                    if not previous_inside:
                        output.append(self._edge_intersection(previous, current, edge_start, edge_end))
                    output.append(current)
                elif previous_inside:
                    output.append(self._edge_intersection(previous, current, edge_start, edge_end))

        if len(output) < 3:
            return None

        return np.array(output, dtype=float)
    
    def _edge_intersection(self, p1, p2, edge_start, edge_end):
            edge_direction = edge_end - edge_start
            segment_direction = p2 - p1

            denominator = PolygonOps.cross_2d(edge_direction, segment_direction)
            t = PolygonOps.cross_2d(edge_direction, edge_start - p1) / denominator

            return p1 + t * segment_direction
    
    def get_intersection_centroid(self, node: "RectRegion") -> GridCell:
        if not self.intersects(node):
            raise Exception("The nodes are not intersecting...")
        
        polygon = self.get_intersection_polygon(node)

        centroid_x, centroid_y = PolygonOps.get_polygon_centroid(polygon)

        center: GridCell = (round(centroid_x), round(centroid_y))

        return center
    
    def edge_cost(self, node, edge_weight: float) -> float:
        if not self.intersects(node):
            raise Exception("The nodes are not intersecting...")
         
        polygon = self.get_intersection_polygon(node)
        area = PolygonOps.get_polygon_area(polygon)

        ref_x, ref_y = PolygonOps.get_polygon_centroid(polygon)

        distance_first = math.hypot(self.center_x - ref_x, self.center_y - ref_y)

        distance_second = math.hypot(node.center_x - ref_x, node.center_y - ref_y)

        return distance_first + distance_second + (edge_weight / area)