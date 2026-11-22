import math
import numpy as np
from scipy import ndimage
from global_planner.rect_region import RectRegion, GridCell

class RectNodeGraphGenerator:
    def __init__(self, obstacle_grid: np.ndarray, alpha: float = 0.99, p_confidence: float = 0.90, gamma: float = 1.05, max_nodes: int = 1000, edge_cost_weight: float = 1.0, dst_thr: float = 0.0, random_seed: int = 20, robot_radius: float = 2.0):
        self.obstacles = obstacle_grid.astype(bool)
        self.free_mask = ~self.obstacles
        
        if dst_thr < 0.0:
            raise ValueError("dst_thr must be non-negative.")
        self.dst_thr = dst_thr

        if robot_radius <= 0.0:
            raise ValueError("robot_radius must be positive.")
        self.robot_radius = robot_radius

        self.obstacle_x, self.obstacle_y = np.nonzero(self.obstacles)

        if self.obstacle_x.size > 0:
            self.nearest_obstacle_index = ndimage.distance_transform_edt(self.free_mask, return_distances=False, return_indices=True)
        else:
            self.nearest_obstacle_index = None

        self.number_of_columns, self.number_of_rows = self.obstacles.shape
        
        if gamma <= 1.0:
            raise ValueError("gamma must be greater than 1 for the node expansion to terminate.")
        self.gamma = gamma
        
        self.termination_threshold = math.ceil(math.log(1.0 - p_confidence) / math.log(alpha) - 1.0)
        self.max_nodes = max_nodes

        self.rng = np.random.default_rng(random_seed)
        
        self.nodes: list[RectRegion] = []

        self.graph: dict[int, dict[int, float]] = {}

        self.edge_cost_weight = edge_cost_weight

    def _cell_of_point(self, q_x: float, q_y: float) -> GridCell:
        x = min(max(int(round(q_x)), 0), self.number_of_columns - 1)
        y = min(max(int(round(q_y)), 0), self.number_of_rows - 1)
        cell:GridCell = (x, y)
        return cell

    def _point_in_obstacle(self, x: float, y: float) -> bool:
        x, y = self._cell_of_point(x, y)
        return bool(self.obstacles[x, y])
    
    def _point_in_any_node(self, x: float, y: float) -> bool:
        return any(node.contains((x, y)) for node in self.nodes)
    
    def _sample_random_point(self):
        failure_num = 0
        while True:
            x = float(self.rng.uniform(-0.5, self.number_of_columns - 0.5))
            y = float(self.rng.uniform(-0.5, self.number_of_rows - 0.5))

            if self._point_in_obstacle(x, y):
                continue

            _, clearance = self._get_nearest_obstacle_point(x, y)
            if clearance < self.robot_radius or self._point_in_any_node(x, y):
                failure_num += 1
                if failure_num >= self.termination_threshold:
                    return None, failure_num
                continue

            return (x, y), failure_num
        
        
    def _get_nearest_obstacle_point(self, x: float, y: float):
        candidates: list[tuple[np.ndarray, float]] = []
        if self.nearest_obstacle_index is not None:

            cell_x, cell_y = self._cell_of_point(x, y)
            radius = 2
            x_lo, x_hi = max(cell_x - radius, 0), min(cell_x + radius, self.number_of_columns - 1)
            y_lo, y_hi = max(cell_y - radius, 0), min(cell_y + radius, self.number_of_rows - 1)

            window = self.nearest_obstacle_index[:, x_lo:x_hi + 1, y_lo:y_hi + 1]
            candidate_x = window[0].ravel()
            candidate_y = window[1].ravel()

            closest_x_on_cells = np.clip(x, candidate_x - 0.5, candidate_x + 0.5)

            closest_y_on_cells = np.clip(y, candidate_y - 0.5, candidate_y + 0.5)

            distance_squared = ((closest_x_on_cells - x) ** 2 + (closest_y_on_cells - y) ** 2)

            closest_index = int(np.argmin(distance_squared))

            closest_obstacle_point = np.array([closest_x_on_cells[closest_index], closest_y_on_cells[closest_index]], dtype=float)

            closest_obstacle_distance = math.sqrt(float(distance_squared[closest_index]))

            candidates.append((closest_obstacle_point, closest_obstacle_distance))

        boundary_candidates = [
            (np.array([-0.5, y]), x + 0.5,),
            (np.array([self.number_of_columns - 0.5, y]), self.number_of_columns - 0.5 - x),
            (np.array([x, -0.5]), y + 0.5),
            (np.array([x, self.number_of_rows - 0.5]), self.number_of_rows - 0.5 - y)
        ]

        candidates.extend(boundary_candidates)

        closest_point, minimum_distance = min(candidates,key=lambda candidate: candidate[1])

        if self.dst_thr > 0.0:
            if minimum_distance > 1e-12:
                query = np.array([x, y], dtype=float)
                closest_point = closest_point + (self.dst_thr / minimum_distance) * (query - closest_point)
            minimum_distance = minimum_distance - self.dst_thr

        return closest_point, minimum_distance
    
    def _generate_square_region(self, rand_point: tuple[float, float]) -> RectRegion:
        x, y = rand_point
        rand_point = np.array([x, y], dtype=float)
        obstacle_point, minimum_distance = (self._get_nearest_obstacle_point(x, y))
        direction_to_obstacle = (obstacle_point - rand_point)
        direction_norm = np.linalg.norm(direction_to_obstacle)
        if direction_norm <= 1e-12:
            raise RuntimeError("Could not determine the nearest obstacle direction.")
        direction_to_obstacle /= direction_norm
        edge_direction = np.array([-direction_to_obstacle[1], direction_to_obstacle[0]])
        theta = math.atan2(edge_direction[1], edge_direction[0])
        half_side = (minimum_distance / math.sqrt(2.0))
        half_side *= 0.999
        rectangle = RectRegion(
            center_x = x,
            center_y = y,
            theta = theta,
            half_width = half_side,
            half_height = half_side
        )
        return rectangle

    def _rect_in_arena(self, rectangle: RectRegion) -> bool:
        vertices = rectangle.vertices

        # Arena walls are obstacles too: inflate them inward by dst_thr.
        return (
            vertices[:, 0].min() >= -0.5 + self.dst_thr
            and vertices[:, 0].max() <= self.number_of_columns - 0.5 - self.dst_thr
            and vertices[:, 1].min() >= -0.5 + self.dst_thr
            and vertices[:, 1].max() <= self.number_of_rows - 0.5 - self.dst_thr
        )

    def _rect_hits_obstacle(self, rectangle: RectRegion, tolerance: float = 1e-9) -> bool:
        if len(self.obstacle_x) == 0:
            return False

        obstacle_centers = np.stack([self.obstacle_x, self.obstacle_y], axis=1).astype(float)

        # Separating Axis Theorem between the oriented rectangle and every
        # axis-aligned unit obstacle cell, vectorized over the cells.
        test_axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), rectangle.axis_1, rectangle.axis_2]

        overlapping = np.ones(len(obstacle_centers), dtype=bool)

        for axis in test_axes:
            rectangle_radius = (rectangle.half_width * abs(float(axis @ rectangle.axis_1)) + rectangle.half_height * abs(float(axis @ rectangle.axis_2)))

            cell_radius = 0.5 * (abs(axis[0]) + abs(axis[1])) + self.dst_thr

            center_distances = np.abs((obstacle_centers - rectangle.center) @ axis)

            overlapping &= center_distances < rectangle_radius + cell_radius - tolerance

            if not overlapping.any():
                return False

        return True

    def _rect_is_free(self, rectangle: RectRegion) -> bool:
        return self._rect_in_arena(rectangle) and not self._rect_hits_obstacle(rectangle)

    def _expand_face(self, node: RectRegion, axis_index: int, sign: int) -> RectRegion:
        axis = node.axis_1 if axis_index == 1 else node.axis_2
        while True:
            half = node.half_width if axis_index == 1 else node.half_height
            step = half * (self.gamma - 1.0)
            offset = sign * 0.5 * step

            node.center_x += offset * axis[0]
            node.center_y += offset * axis[1]
            if axis_index == 1:
                node.half_width += 0.5 * step
            else:
                node.half_height += 0.5 * step

            if not self._rect_is_free(node):
                node.center_x -= offset * axis[0]
                node.center_y -= offset * axis[1]
                if axis_index == 1:
                    node.half_width -= 0.5 * step
                else:
                    node.half_height -= 0.5 * step
                return node

    def _expand_along_axis_asymmetric(self, node: RectRegion, axis_index: int) -> RectRegion:
        node = self._expand_face(node, axis_index, sign=+1)
        node = self._expand_face(node, axis_index, sign=-1)
        return node

    def _expand_along_axis_symmetric(self, node: RectRegion, axis_index: int) -> RectRegion:
        while True:
            if axis_index == 1:
                node.half_width *= self.gamma
            else:
                node.half_height *= self.gamma

            if not self._rect_is_free(node):
                if axis_index == 1:
                    node.half_width /= self.gamma
                else:
                    node.half_height /= self.gamma
                return node

    def _expand(self, square_node: RectRegion) -> RectRegion:
        node = self._expand_along_axis_symmetric(square_node, axis_index=1)
        node = self._expand_along_axis_symmetric(node, axis_index=2)
        return node

    def _insert_node(self, node: RectRegion):
        node_id = len(self.nodes)

        self.nodes.append(node)
        self.graph[node_id] = {}

        for old_node_id, old_rectangle in enumerate(self.nodes[:-1]):
            if not node.intersects(old_rectangle):
                continue

            cost = node.edge_cost(old_rectangle, self.edge_cost_weight)

            if not math.isfinite(cost):
                continue

            self.graph[node_id][old_node_id] = cost
            self.graph[old_node_id][node_id] = cost


    def generate(self):
        for _ in range(self.max_nodes):
            rand_point, _ = self._sample_random_point()

            if rand_point is None:
                break

            square_node = self._generate_square_region(rand_point)
            node = self._expand(square_node)

            if min(node.half_width, node.half_height) < self.robot_radius:
                continue

            self._insert_node(node)

        return {
            "nodes": self.nodes,
            "graph": self.graph
        }
             

        
