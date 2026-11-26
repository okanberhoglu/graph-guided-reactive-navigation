from global_motion_planner.rect_region import GridCell, RectRegion
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Patch, Polygon


class RobotVisualization:

    def __init__(self, fig: Figure, ax_main: Axes, ax_local: Axes, obstacle_grid: np.ndarray, robot_pose: tuple[float, float, float], robot_radius: float, sensor_radius: float,) -> None:
        self.fig = fig
        self.ax_main = ax_main
        self.ax_local = ax_local
        self.obstacle_grid = obstacle_grid
        self.robot_radius = robot_radius
        self.sensor_radius = sensor_radius

        self.visible_mask = None

        robot_x, robot_y, robot_theta = robot_pose

        # Robot body and heading on the main plot.
        self.robot_circle, self.robot_arrow = Visualizer.draw_robot(ax=ax_main, robot_pose=robot_pose, robot_radius=robot_radius)

        # Gray transparent sensor range around the robot.
        self.sensor_circle = Circle((robot_x, robot_y), sensor_radius, facecolor="gray", edgecolor="dimgray", alpha=0.3, zorder=4)

        ax_main.add_patch(self.sensor_circle)

        # Robot-centered local occupancy view.
        local_map = Visualizer.extract_local_map( obstacle_grid=obstacle_grid, robot_pose=robot_pose, sensor_radius=sensor_radius)

        half_size = local_map.shape[0] / 2.0

        self.local_image = ax_local.imshow(local_map, cmap="gray_r", origin="lower", interpolation="none", vmin=0.0, vmax=1.0,extent=(-half_size, half_size, -half_size, half_size), zorder=0)

        # The robot stays at the center of the local view; only
        # the map content and the heading arrow change.
        self.local_robot_circle = Circle((0.0, 0.0), robot_radius, color="blue", alpha=0.6, zorder=6)

        ax_local.add_patch(self.local_robot_circle)

        self.local_robot_arrow = Visualizer._draw_heading_arrow(ax=ax_local, robot_pose=(0.0, 0.0, robot_theta), robot_radius=robot_radius)

        # Control-loop artists, created on first use.
        self.vector_field_quiver = None
        self.trajectory_line = None
        self.rollout_lines: list = []

        ax_local.set_aspect("equal")
        ax_local.set_xlabel("Local x [cells]")
        ax_local.set_ylabel("Local y [cells]")

        ax_local.set_title(f"Local Occupancy Grid Map "f"{sensor_radius:g} cells")

    def update(self, robot_pose: tuple[float, float, float], pause: float | None = 0.01):

        robot_x, robot_y, robot_theta = robot_pose

        self.robot_circle.center = (robot_x, robot_y)
        self.sensor_circle.center = (robot_x, robot_y)

        if self.robot_arrow is not None:
            self.robot_arrow.remove()

        self.robot_arrow = Visualizer._draw_heading_arrow(ax=self.ax_main, robot_pose=robot_pose, robot_radius=self.robot_radius)

        local_map = Visualizer.extract_local_map( obstacle_grid=self.obstacle_grid, robot_pose=robot_pose, sensor_radius=self.sensor_radius, visible_mask=self.visible_mask)

        self.local_image.set_data(local_map)

        if self.local_robot_arrow is not None:
            self.local_robot_arrow.remove()

        self.local_robot_arrow = Visualizer._draw_heading_arrow(ax=self.ax_local, robot_pose=(0.0, 0.0, robot_theta), robot_radius=self.robot_radius)

        if pause is not None:
            plt.pause(pause)

    def update_vector_field(self, points: np.ndarray, vectors: np.ndarray, arrow_length: float = 2.5,) -> None:

        if self.vector_field_quiver is not None:
            self.vector_field_quiver.remove()
            self.vector_field_quiver = None

        if len(points) == 0:
            return

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)

        arrows = vectors / norms * arrow_length

        self.vector_field_quiver = self.ax_main.quiver(
            points[:, 0],
            points[:, 1],
            arrows[:, 0],
            arrows[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0025,
            color="darkgreen",
            alpha=0.8,
            zorder=5,
        )

    def update_trajectory(self, trajectory: np.ndarray | list,) -> None:
        if self.trajectory_line is None:
            (self.trajectory_line,) = self.ax_main.plot([], [], "b-", linewidth=2.0, alpha=0.8, zorder=5)

        trajectory = np.asarray(trajectory)

        self.trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])

    def update_rollouts(self, rollouts_xy: list) -> None:
        for line in self.rollout_lines:
            line.remove()

        self.rollout_lines = []

        for idx, rollout in enumerate(rollouts_xy):
            rollout = np.asarray(rollout)

            (line,) = self.ax_main.plot(
                rollout[:, 0],
                rollout[:, 1],
                color="cyan",
                linestyle="--",
                linewidth=1.0,
                alpha=max(0.1, 0.3 - 0.1 * idx),
                zorder=5,
            )

            self.rollout_lines.append(line)


class Visualizer:
    
    @staticmethod
    def visualize_graph( obstacle_grid: np.ndarray, nodes: list[RectRegion], temp_goals: list[GridCell] | None = None, goal: tuple[float, float] | None = None, start: tuple[float, float] | None = None, robot_pose: tuple[float, float, float] | None = None, robot_radius: float = 2.5, sensor_radius: float = 12.0, block: bool = True, title: str | None = None) -> tuple[Figure, Axes, RobotVisualization | None]:
        columns, rows = obstacle_grid.shape

        if robot_pose is not None:
            fig, (ax, ax_local) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2.4, 1.0]})
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax_local = None

        ax.imshow(
            obstacle_grid.T,
            cmap="gray_r",
            origin="lower",
            interpolation="none",
            extent=(
                -0.5,
                columns - 0.5,
                -0.5,
                rows - 0.5,
            ),
            zorder=0,
        )

        Visualizer.draw_nodes(ax=ax, nodes=nodes)

        if temp_goals:
            Visualizer.draw_temp_goals(ax=ax, temp_goals=temp_goals)

        if start is not None:
            Visualizer.draw_start(ax=ax, start=start)

        if goal is not None:
            Visualizer.draw_goal(ax=ax, goal=goal)

        robot_visualization = None

        if robot_pose is not None:
            robot_visualization = RobotVisualization(fig=fig, ax_main=ax, ax_local=ax_local, obstacle_grid=obstacle_grid, robot_pose=robot_pose, robot_radius=robot_radius, sensor_radius=sensor_radius)

        ax.set_xlim(-0.5, columns - 0.5)

        ax.set_ylim(-0.5, rows - 0.5)

        ax.set_aspect("equal")
        ax.set_xlabel("Column, x")
        ax.set_ylabel("Row, y")

        if title is None:
            title = (f"MPC Graph Oriented Rectangular Regions — "f"{len(nodes)} Nodes")

        ax.set_title(title)

        legend_items = [
            Patch(
                facecolor="black",
                edgecolor="black",
                label="Obstacle",
            ),
            Patch(
                facecolor="gray",
                edgecolor="blue",
                alpha=0.25,
                linewidth=1.5,
                label="Rectangular node",
            ),
        ]

        if temp_goals:
            legend_items.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color="blue",
                    markersize=6,
                    label="Temp goal",
                )
            )

        if start is not None:
            legend_items.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color="red",
                    markersize=8,
                    label="Start",
                )
            )

        if goal is not None:
            legend_items.append(
                plt.Line2D(
                    [],
                    [],
                    marker="x",
                    linestyle="none",
                    color="green",
                    markeredgewidth=2.5,
                    markersize=9,
                    label="Goal",
                )
            )

        if robot_pose is not None:
            legend_items.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color="blue",
                    alpha=0.6,
                    markersize=10,
                    label="Robot",
                )
            )

            legend_items.append(
                Patch(
                    facecolor="gray",
                    edgecolor="dimgray",
                    alpha=0.3,
                    label="Sensor range",
                )
            )

        ax.legend(
            handles=legend_items,
            loc="upper right",
        )

        plt.tight_layout()
        plt.show(block=block)

        return fig, ax, robot_visualization

    @staticmethod
    def draw_nodes(ax: Axes, nodes: list[RectRegion]) -> list[Polygon]:
        color_map = plt.get_cmap("tab20")

        patches: list[Polygon] = []

        for node_id, node in enumerate(nodes):
            node_color = color_map(node_id % color_map.N)

            rectangle_polygon = Polygon(
                node.vertices,
                closed=True,
                facecolor=node_color,
                edgecolor="blue",
                alpha=0.16,
                linewidth=1.5,
                zorder=2,
            )

            ax.add_patch(rectangle_polygon)

            patches.append(rectangle_polygon)

        return patches

    @staticmethod
    def draw_temp_goals(ax: Axes, temp_goals: list[GridCell]) -> list:
        if not temp_goals:
            return []

        temp_goal_x = [cell[0] for cell in temp_goals]
        temp_goal_y = [cell[1] for cell in temp_goals]

        line, = ax.plot(
            temp_goal_x,
            temp_goal_y,
            marker="o",
            markersize=6,
            color="blue",
            linestyle="none",
            label="Temp goal",
            zorder=5,
        )

        return [line]

    @staticmethod
    def draw_start(ax: Axes, start: tuple[float, float]):
        line, = ax.plot(
            start[0],
            start[1],
            marker="o",
            markersize=9,
            color="red",
            linestyle="none",
            label="Start",
            zorder=5,
        )

        return line

    @staticmethod
    def draw_goal(ax: Axes, goal: tuple[float, float]):
        line, = ax.plot(
            goal[0],
            goal[1],
            marker="x",
            markersize=11,
            markeredgewidth=3,
            color="green",
            linestyle="none",
            label="Goal",
            zorder=5,
        )

        return line

    @staticmethod
    def draw_robot(ax: Axes, robot_pose: tuple[float, float, float], robot_radius: float) -> tuple[Circle, FancyArrow]:
        robot_x, robot_y, _ = robot_pose

        robot_circle = Circle(
            (robot_x, robot_y),
            robot_radius,
            color="blue",
            alpha=0.6,
            zorder=6,
        )

        ax.add_patch(
            robot_circle
        )

        robot_arrow = Visualizer._draw_heading_arrow(
            ax=ax,
            robot_pose=robot_pose,
            robot_radius=robot_radius,
        )

        return robot_circle, robot_arrow

    @staticmethod
    def draw_edt(ax: Axes, clearance: np.ndarray, extent: tuple[float, float, float, float], clim: tuple[float, float], robot_pose: tuple[float, float, float], robot_radius: float, gradient: tuple[float, float], footprint: tuple[float, float, float] | None = None, gradient_origin: tuple[float, float] | None = None) -> AxesImage:
        """Draw the controller's distance window, clearance indexed [x, y] (< 0 = footprint in collision), in map cells.

        footprint (front, rear, half_width), measured along the heading from the robot pose, draws a rectangle instead of
        the circle of robot_radius; the gradient arrow starts at gradient_origin (default: the robot)."""
        ax.clear()

        # clearance is indexed [x, y]; Matplotlib images use [row, column]
        image = ax.imshow(clearance.T, origin="lower", extent=extent, interpolation="nearest", aspect="equal", vmin=clim[0], vmax=clim[1])

        # zero clearance is where the circular footprint touches an obstacle
        columns, rows = clearance.shape
        if min(columns, rows) > 1 and clearance.min() < 0.0 < clearance.max():
            x_min, x_max, y_min, y_max = extent
            x_coords = x_min + (np.arange(columns) + 0.5) * (x_max - x_min) / columns
            y_coords = y_min + (np.arange(rows) + 0.5) * (y_max - y_min) / rows
            ax.contour(x_coords, y_coords, clearance.T, levels=[0.0], colors="white", linewidths=1.5)

        robot_x, robot_y, robot_theta = robot_pose
        ax.plot(robot_x, robot_y, "o", color="red", markersize=6, label="Robot")

        if footprint is None:
            ax.add_patch(Circle((robot_x, robot_y), robot_radius, fill=False, edgecolor="red", linewidth=2))
            arrow_size = robot_radius
        else:
            front, rear, half_width = footprint
            cos_theta, sin_theta = np.cos(robot_theta), np.sin(robot_theta)
            corners = [(front, half_width), (-rear, half_width), (-rear, -half_width), (front, -half_width)]
            outline = [(robot_x + cos_theta * u - sin_theta * v, robot_y + sin_theta * u + cos_theta * v) for u, v in corners]
            ax.add_patch(Polygon(outline, closed=True, fill=False, edgecolor="red", linewidth=2))
            arrow_size = half_width

        Visualizer._draw_heading_arrow(ax=ax, robot_pose=robot_pose, robot_radius=arrow_size)

        # the gradient points away from the nearest obstacle; a unit gradient is drawn a tenth of the window long
        origin_x, origin_y = gradient_origin if gradient_origin is not None else (robot_x, robot_y)
        arrow_scale = 10.0 / (extent[1] - extent[0])
        ax.quiver(origin_x, origin_y, gradient[0], gradient[1], angles="xy", scale_units="xy", scale=arrow_scale, color="magenta", width=0.006, zorder=8, label="Clearance gradient")

        ax.legend(loc="upper right")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Column, x")
        ax.set_ylabel("Row, y")
        ax.set_title("Controller EDT")

        return image

    @staticmethod
    def extract_local_map(obstacle_grid: np.ndarray, robot_pose: tuple[float, float, float], sensor_radius: float, visible_mask: np.ndarray | None = None) -> np.ndarray:
        columns, rows = obstacle_grid.shape

        robot_x, robot_y, _ = robot_pose

        center_column = int(round(robot_x))
        center_row = int(round(robot_y))

        radius_cells = int(np.ceil(sensor_radius))
        size = 2 * radius_cells + 1

        local_map = np.full((size, size), 0.5, dtype=float)

        # Map window overlapping the sensor square, clipped to
        # the grid borders.
        row_start = max(0, center_row - radius_cells)
        row_end = min(rows, center_row + radius_cells + 1)
        column_start = max(0, center_column - radius_cells)
        column_end = min(columns, center_column + radius_cells + 1)

        if row_start < row_end and column_start < column_end:
            local_row_start = row_start - (center_row - radius_cells)
            local_column_start = column_start - (center_column - radius_cells)

            # obstacle_grid is indexed [x, y] (column, row); local_map
            # is laid out [row, column] for display, so the window is
            # transposed on the way in.
            window = obstacle_grid[column_start:column_end, row_start:row_end].T

            local_map[local_row_start:local_row_start + window.shape[0], local_column_start:local_column_start + window.shape[1]] = window.astype(float)

            if visible_mask is not None:
                local_visible = np.zeros((size, size), dtype=bool)
                local_visible[local_row_start:local_row_start + window.shape[0], local_column_start:local_column_start + window.shape[1]] = visible_mask[column_start:column_end, row_start:row_end].T
                local_map[~local_visible] = 0.5

        # Hide every cell outside the circular sensor range.
        local_rows, local_columns = np.mgrid[ 0:size,0:size]

        distance_squared = ((local_rows - radius_cells) ** 2 + (local_columns - radius_cells) ** 2)

        local_map[distance_squared > sensor_radius**2] = 0.5

        return local_map

    @staticmethod
    def local_free_points(obstacle_grid: np.ndarray, robot_pose: tuple[float, float, float], sensor_radius: float, stride: int = 3, visible_mask: np.ndarray | None = None) -> np.ndarray:
        columns, rows = obstacle_grid.shape

        robot_x, robot_y, _ = robot_pose

        # obstacle_grid is indexed [x, y], so the coordinate grids
        # are built in that same order to line up with it.
        column_coordinates, row_coordinates = np.mgrid[0:columns, 0:rows]

        distance_squared = ((column_coordinates - robot_x) ** 2 + (row_coordinates - robot_y) ** 2)

        mask = ((distance_squared <= sensor_radius**2) & ~obstacle_grid)

        if visible_mask is not None:
            mask &= visible_mask

        if stride > 1:
            mask &= ((row_coordinates % stride == 0) & (column_coordinates % stride == 0))

        return np.stack(
            [
                column_coordinates[mask],
                row_coordinates[mask],
            ],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def visibility_mask(obstacle_grid: np.ndarray, robot_pose: tuple[float, float, float], sensor_radius: float, ray_step: float = 0.5) -> np.ndarray:
        columns, rows = obstacle_grid.shape

        robot_x, robot_y, _ = robot_pose

        # obstacle_grid is indexed [x, y], so the coordinate grids
        # are built in that same order to line up with it.
        column_coordinates, row_coordinates = np.mgrid[
            0:columns,
            0:rows,
        ]

        delta_x = column_coordinates - robot_x
        delta_y = row_coordinates - robot_y
        distance = np.sqrt(delta_x**2 + delta_y**2)

        inside = distance <= sensor_radius

        target_columns = column_coordinates[inside]
        target_rows = row_coordinates[inside]
        target_distance = distance[inside]

        distance_safe = np.maximum(target_distance, 1e-6)
        unit_x = (target_columns - robot_x) / distance_safe
        unit_y = (target_rows - robot_y) / distance_safe

        steps = np.arange(ray_step, sensor_radius, ray_step)

        sample_x = robot_x + unit_x[:, None] * steps[None, :]
        sample_y = robot_y + unit_y[:, None] * steps[None, :]

        sample_columns = np.clip(np.round(sample_x).astype(int), 0, columns - 1)
        sample_rows = np.clip(np.round(sample_y).astype(int), 0, rows - 1)

        occupied = obstacle_grid[sample_columns, sample_rows]

        # A sample blocks the ray only if it lies before the target
        # cell and is not the target cell itself (so an occupied
        # target does not occlude itself).
        before_target = steps[None, :] < (target_distance[:, None] - 0.25)
        same_cell = ((sample_columns == target_columns[:, None]) & (sample_rows == target_rows[:, None]))

        blocked = (occupied & before_target & ~same_cell).any(axis=1)

        visible = np.zeros_like(obstacle_grid, dtype=bool)
        visible[target_columns, target_rows] = ~blocked

        return visible

    @staticmethod
    def _draw_heading_arrow(ax: Axes, robot_pose: tuple[float, float, float], robot_radius: float) -> FancyArrow:
        robot_x, robot_y, robot_theta = robot_pose

        arrow_length = robot_radius * 1.5

        return ax.arrow(
            robot_x,
            robot_y,
            arrow_length * np.cos(robot_theta),
            arrow_length * np.sin(robot_theta),
            head_width=robot_radius * 0.6,
            head_length=robot_radius * 0.4,
            fc="darkblue",
            ec="darkblue",
            linewidth=2,
            zorder=7,
        )