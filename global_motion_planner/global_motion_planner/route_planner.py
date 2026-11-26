from global_motion_planner.rect_region import RectRegion, GridCell
import math
import heapq

class DijkstraRoutePlanner:
    def __init__(self, nodes: list[RectRegion], graph: dict[int, dict[int, float]]):
            self.nodes = nodes
            self.graph = graph
    
            self.start_node_id_list: list[int] = []
            self.goal_node_id_list: list[int] = []

    def _find_node(self, cell: GridCell) -> list[int]:
            id_list: list[int] = []
    
            for id, node in enumerate(self.nodes):
                if node.contains(cell):
                    id_list.append(id)
    
            return id_list

    def get_temp_goal_points(self) -> list[tuple[float, float]]:
            route_nodes = [self.nodes[node_id] for node_id in self.route_node_ids]
            points: list[tuple[float, float]] = []
    
            for i in range(len(self.route_node_ids) - 1):
                point = route_nodes[i].get_intersection_centroid(route_nodes[i + 1])
                points.append(point)
    
            points.append((float(self.goal_point[0]), float(self.goal_point[1])))
    
            return points

    @staticmethod
    def _segment_interval_in_node(node: RectRegion, p: tuple[float, float], q: tuple[float, float]): 
            axis_1 = node.axis_1
            axis_2 = node.axis_2
    
            relative_x = p[0] - node.center_x
            relative_y = p[1] - node.center_y
    
            direction_x = q[0] - p[0]
            direction_y = q[1] - p[1]
    
            t_low, t_high = 0.0, 1.0
    
            for axis, half_extent in ((axis_1, node.half_width), (axis_2, node.half_height)):
                offset = relative_x * axis[0] + relative_y * axis[1]
                slope = direction_x * axis[0] + direction_y * axis[1]
    
                if abs(slope) < 1e-12:
                    if abs(offset) > half_extent:
                        return None
                    continue
    
                t_enter = (-half_extent - offset) / slope
                t_exit = (half_extent - offset) / slope
    
                if t_enter > t_exit:
                    t_enter, t_exit = t_exit, t_enter
    
                t_low = max(t_low, t_enter)
                t_high = min(t_high, t_exit)
    
                if t_low > t_high:
                    return None
    
            return t_low, t_high
        
    def _segment_cover(self, p: tuple[float, float], q: tuple[float, float]) -> list[int] | None:
            # Ordered node ids that together cover the segment p -> q, or None
            # if some part of it is outside every node. Greedy: always take the
            # node that carries the segment furthest, so the cover has the
            # fewest nodes. Every node is obstacle free, so any node may be
            # used, not only the ones on the route.
            intervals = []

            for node_id, node in enumerate(self.nodes):
                interval = self._segment_interval_in_node(node, p, q)

                if interval is not None:
                    intervals.append((interval[0], interval[1], node_id))

            intervals.sort()

            epsilon = 1e-9
            cover: list[int] = []
            covered_up_to = 0.0
            k = 0

            while covered_up_to < 1.0 - epsilon:
                best_end, best_id = covered_up_to, None

                while k < len(intervals) and intervals[k][0] <= covered_up_to + epsilon:
                    if intervals[k][1] > best_end:
                        best_end, best_id = intervals[k][1], intervals[k][2]
                    k += 1

                if best_id is None:
                    return None

                cover.append(best_id)
                covered_up_to = best_end

            return cover

    def get_pruned_temp_goal_points(self) -> list[tuple[float, float]]:
            waypoints = [(float(self.start_point[0]), float(self.start_point[1]))]
            waypoints.extend(self.get_temp_goal_points())

            pruned: list[tuple[float, float]] = []
            # Nodes the pruned path runs through, in driving order. A shortcut
            # can pass through nodes that are not on the Dijkstra route.
            pruned_route: list[int] = []
            current_index = 0

            while current_index < len(waypoints) - 1:
                next_index = len(waypoints) - 1
                cover = self._segment_cover(waypoints[current_index], waypoints[next_index])

                while cover is None and next_index > current_index + 1:
                    next_index -= 1
                    cover = self._segment_cover(waypoints[current_index], waypoints[next_index])

                # Neighbouring waypoints both lie in route node current_index;
                # this only matters if rounding a centroid moved it outside.
                if cover is None:
                    cover = [self.route_node_ids[current_index]]

                for node_id in cover:
                    if not pruned_route or pruned_route[-1] != node_id:
                        pruned_route.append(node_id)

                pruned.append(waypoints[next_index])
                current_index = next_index

            self.pruned_route_node_ids = pruned_route

            return pruned

    def get_direct_route_to_goal(self, position):
        p = (float(position[0]), float(position[1]))
        q = (float(self.goal_point[0]), float(self.goal_point[1]))
        return self._segment_cover(p, q)

    def _reconstruct_route(self, came_from: dict[int, int], reached_goal_id: int) -> list[int]:
            route: list[int] = [reached_goal_id]
    
            while route[-1] in came_from:
                route.append(came_from[route[-1]])
    
            route.reverse()
    
            return route

    def _dijkstra_search(self) -> dict[int, int]:
            if not self.start_node_id_list:
                raise ValueError("The start position is not covered by any node.")

            if not self.goal_node_id_list:
                raise ValueError("The goal position is not covered by any node.")

            goal_set = set(self.goal_node_id_list)

            cost_to_goal: dict[int, float] = {node_id: math.inf for node_id in self.graph}

            policy: dict[int, int] = {}

            open_queue: list[tuple[float, int]] = []

            for goal_node_id in goal_set:
                cost_to_goal[goal_node_id] = 0.0
                heapq.heappush(open_queue, (0.0, goal_node_id))

            settled_nodes: set[int] = set()

            while open_queue:
                current_cost, current_node_id = heapq.heappop(open_queue)

                if current_node_id in settled_nodes:
                    continue

                settled_nodes.add(current_node_id)

                for neighbor_id, edge_cost in self.graph.get(current_node_id, {}).items():
                    if neighbor_id in settled_nodes:
                        continue

                    if not math.isfinite(edge_cost):
                        continue

                    tentative_cost = current_cost + edge_cost

                    if tentative_cost < cost_to_goal.get(neighbor_id, math.inf):
                        cost_to_goal[neighbor_id] = tentative_cost

                        policy[neighbor_id] = current_node_id

                        heapq.heappush(open_queue, (tentative_cost, neighbor_id))

            self.cost_to_goal = cost_to_goal
            self.policy = policy

            return policy

    def find_route(self, start: GridCell, goal: GridCell) -> list[int]:
            self.route_node_ids: list[int] = []

            self.start_point = start
            self.goal_point = goal

            self.start_node_id_list = self._find_node(start)
            self.goal_node_id_list = self._find_node(goal)

            self._dijkstra_search()

            goal_set = set(self.goal_node_id_list)

            # The start point can sit in several overlapping nodes. Take the
            # one with the cheapest route to the goal and, among equal ones,
            # the largest, since a bigger region leaves the MPC more room
            # (thesis Sec. 3.3: the CurrentNode ensuring the minimum cost
            # route according to eq. 3.4).
            current_node_id = min(self.start_node_id_list, key=lambda node_id: (self.cost_to_goal[node_id], -self.nodes[node_id].area))

            if not math.isfinite(self.cost_to_goal[current_node_id]):
                raise RuntimeError("No route exists between the start and goal positions.")

            self.route_node_ids.append(current_node_id)

            # Walking Po from the start node is exactly Po.Next chained
            # until the goal node, and it terminates because the cost to
            # the goal strictly decreases at every hop.
            while current_node_id not in goal_set:
                current_node_id = self.policy[current_node_id]

                self.route_node_ids.append(current_node_id)

            return self.route_node_ids

    def update_route(self, position: GridCell) -> list[int]:
            containing_node_ids = self._find_node(position)

            if not containing_node_ids:
                raise ValueError("The robot position is not covered by any node.")

            goal_set = set(self.goal_node_id_list)

            # Same choice as in find_route: the cheapest node to the goal,
            # and the largest one among equal costs.
            current_node_id = min(containing_node_ids, key=lambda node_id: (self.cost_to_goal[node_id], -self.nodes[node_id].area))

            if not math.isfinite(self.cost_to_goal[current_node_id]):
                raise RuntimeError("No route exists between the robot position and the goal.")

            route_node_ids: list[int] = [current_node_id]

            while current_node_id not in goal_set:
                current_node_id = self.policy[current_node_id]

                route_node_ids.append(current_node_id)

            # The robot position becomes the start point, which the pruned
            # temp goals start from.
            self.start_point = position
            self.start_node_id_list = containing_node_ids
            self.route_node_ids = route_node_ids

            return self.route_node_ids