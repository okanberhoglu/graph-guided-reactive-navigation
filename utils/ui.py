import os
from global_planner.rect_region import GridCell, RectRegion
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Patch, Polygon
from matplotlib.transforms import Bbox


class RobotVisualization:

    def __init__(self, fig: Figure, ax_main: Axes, ax_local: Axes, obstacle_grid: np.ndarray, robot_pose: tuple[float, float, float], robot_radius: float, sensor_radius: float, ax_edt: Axes | None = None, save_global_map: bool = False, save_local_map: bool = False, save_edt_map: bool = False, footprint: tuple[float, float, float] | None = None) -> None:
        self.fig = fig
        self.ax_main = ax_main
        self.ax_local = ax_local
        self.ax_edt = ax_edt
        self.edt_colorbar = None

        # which panels save_plots writes to image files
        self.save_global_map = save_global_map
        self.save_local_map = save_local_map
        self.save_edt_map = save_edt_map
        self.obstacle_grid = obstacle_grid
        self.robot_radius = robot_radius
        self.sensor_radius = sensor_radius

        # car: (front, rear, half width) of its box along the heading, drawn instead of the disc of robot_radius
        self.footprint = footprint
        self.arrow_size = footprint[2] if footprint is not None else robot_radius

        self.visible_mask = Visualizer.visibility_mask(obstacle_grid, robot_pose, sensor_radius)

        robot_x, robot_y, robot_theta = robot_pose[:3]

        self.robot_patch, self.robot_arrow = Visualizer.draw_robot(ax=ax_main, robot_pose=robot_pose, robot_radius=robot_radius, footprint=footprint)
        
        self.sensor_circle = Circle((robot_x, robot_y), sensor_radius, facecolor="gray", edgecolor="dimgray", alpha=0.3, zorder=4)

        ax_main.add_patch(self.sensor_circle)

        local_map = Visualizer.extract_local_map( obstacle_grid=obstacle_grid, robot_pose=robot_pose, sensor_radius=sensor_radius, visible_mask=self.visible_mask)

        half_size = local_map.shape[0] / 2.0

        offset_x = round(robot_x) - robot_x
        offset_y = round(robot_y) - robot_y
        self.local_image = ax_local.imshow(local_map, cmap="gray_r", origin="lower", interpolation="none", vmin=0.0, vmax=1.0, extent=(offset_x - half_size, offset_x + half_size, offset_y - half_size, offset_y + half_size), zorder=0)
        ax_local.set_xlim(-half_size, half_size)
        ax_local.set_ylim(-half_size, half_size)

        # the local map is centred on the robot, so it is drawn at the origin
        self.local_robot_patch, self.local_robot_arrow = Visualizer.draw_robot(ax=ax_local, robot_pose=(0.0, 0.0, robot_theta), robot_radius=robot_radius, footprint=footprint)

        self.vector_field_quiver = None
        self.kernel_image = None
        self.kernel_markers = None
        self.kernel_quiver = None
        self.trajectory_line = None
        self.rollout_lines: list = []
        self.temp_goal_points = None
        self.node_patches: list = []
        self.node_ids = None
        self.moving_obstacle_circles: list = []
        self.moving_obstacle_arrows: list = []

        ax_local.set_aspect("equal")
        ax_local.set_xlabel("Local x [cells]")
        ax_local.set_ylabel("Local y [cells]")

        ax_local.set_title(f"Local Occupancy Grid Map "f"{sensor_radius:g} cells")

        if ax_edt is not None:
            ax_edt.set_title("Euclidean Distance Field")
            ax_edt.set_aspect("equal")
            ax_edt.set_xlabel("Map x")
            ax_edt.set_ylabel("Map y")

    def update(self, robot_pose: tuple[float, float, float], pause: float | None = 0.01):

        robot_x, robot_y, robot_theta = robot_pose[:3]

        if self.footprint is None:
            self.robot_patch.center = (robot_x, robot_y)
        else:
            # the box turns with the heading
            self.robot_patch.set_xy(Visualizer.footprint_outline((robot_x, robot_y, robot_theta), self.footprint))
            self.local_robot_patch.set_xy(Visualizer.footprint_outline((0.0, 0.0, robot_theta), self.footprint))
        self.sensor_circle.center = (robot_x, robot_y)

        if self.robot_arrow is not None:
            self.robot_arrow.remove()

        self.robot_arrow = Visualizer._draw_heading_arrow(ax=self.ax_main, robot_pose=robot_pose, robot_radius=self.arrow_size)

        local_map = Visualizer.extract_local_map( obstacle_grid=self.obstacle_grid, robot_pose=robot_pose, sensor_radius=self.sensor_radius, visible_mask=self.visible_mask)

        self.local_image.set_data(local_map)
        half_size = local_map.shape[0] / 2.0
        offset_x = round(robot_x) - robot_x
        offset_y = round(robot_y) - robot_y
        self.local_image.set_extent((offset_x - half_size, offset_x + half_size, offset_y - half_size, offset_y + half_size))

        if self.local_robot_arrow is not None:
            self.local_robot_arrow.remove()

        self.local_robot_arrow = Visualizer._draw_heading_arrow(ax=self.ax_local, robot_pose=(0.0, 0.0, robot_theta), robot_radius=self.arrow_size)

        if pause is not None:
            plt.pause(pause)

    def process_events(self) -> None:
        """Run the GUI event loop once, for use while blocked outside update()."""
        self.fig.canvas.flush_events()

    def update_vector_field(self, points: np.ndarray, vectors: np.ndarray, arrow_length: float = 2.5) -> None:

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
            width=0.0023,
            color="royalblue",
            alpha=0.8,
            zorder=5,
        )

    def update_kernels(self, centers: np.ndarray, directions: np.ndarray, gamma: float, arrow_length: float = 3.0, resolution: float = 0.5,) -> None:

        for artist in (self.kernel_image, self.kernel_markers, self.kernel_quiver):
            if artist is not None:
                artist.remove()

        self.kernel_image = None
        self.kernel_markers = None
        self.kernel_quiver = None

        if len(centers) == 0:
            return

        # Green shaded area where the kernels act: sum of their Gaussian RBFs exp(-gamma d^2), clipped to 1.
        reach = 3.0 / np.sqrt(gamma)

        x_min, y_min = centers.min(axis=0) - reach
        x_max, y_max = centers.max(axis=0) + reach

        xs = np.arange(x_min, x_max + resolution, resolution)
        ys = np.arange(y_min, y_max + resolution, resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)

        activation = np.zeros_like(grid_x)

        for center_x, center_y in centers:
            activation += np.exp(-gamma * ((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2))

        shade = np.zeros(grid_x.shape + (4,))
        shade[..., :3] = (0.2, 0.6, 0.2)
        shade[..., 3] = 0.45 * np.minimum(activation, 1.0)

        self.kernel_image = self.ax_main.imshow(
            shade,
            origin="lower",
            interpolation="bilinear",
            extent=(xs[0] - resolution / 2, xs[-1] + resolution / 2, ys[0] - resolution / 2, ys[-1] + resolution / 2),
            zorder=4.5,
        )

        # Mean deflection direction of each kernel; only its direction is drawn.
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)

        arrows = directions / norms * arrow_length

        self.kernel_markers = self.ax_main.scatter(
            centers[:, 0],
            centers[:, 1],
            s=15,
            color="darkgreen",
            zorder=8,
        )

        self.kernel_quiver = self.ax_main.quiver(
            centers[:, 0],
            centers[:, 1],
            arrows[:, 0],
            arrows[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            color="darkgreen",
            zorder=8,
        )

    def update_moving_obstacles(self, positions: np.ndarray, radius: float, velocities: np.ndarray | None = None) -> None:
        # Pedestrians are drawn on the main map, which itself only shows the static obstacles.
        # With velocities each one also gets an arrow pointing the way it walks (none while it stands still).
        if len(self.moving_obstacle_circles) != len(positions):
            for circle in self.moving_obstacle_circles:
                circle.remove()

            self.moving_obstacle_circles = []

            for x, y in positions:
                circle = Circle((x, y), radius, facecolor="crimson", edgecolor="darkred", alpha=0.85, zorder=6)
                self.ax_main.add_patch(circle)
                self.moving_obstacle_circles.append(circle)

        for circle, (x, y) in zip(self.moving_obstacle_circles, positions):
            circle.center = (x, y)

        for arrow in self.moving_obstacle_arrows:
            arrow.remove()

        self.moving_obstacle_arrows = []

        if velocities is None:
            return

        for (x, y), (vx, vy) in zip(positions, velocities):
            if np.hypot(vx, vy) < 1e-9:
                continue

            heading = float(np.arctan2(vy, vx))
            arrow = Visualizer._draw_heading_arrow(ax=self.ax_main, robot_pose=(x, y, heading), robot_radius=radius, color="black")
            self.moving_obstacle_arrows.append(arrow)

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

    def update_temp_goals(self, temp_goals) -> None:
        if len(temp_goals) == 0:
            return

        if self.temp_goal_points is None:
            (self.temp_goal_points,) = self.ax_main.plot([], [], marker="o", markersize=6, color="blue", linestyle="none", zorder=5, label="Temp goal")

        points = np.asarray(temp_goals, dtype=float)

        self.temp_goal_points.set_data(points[:, 0], points[:, 1])

    def update_nodes(self, nodes) -> None:
        nodes = list(nodes)

        # same nodes as last time: keep the patches already on the map
        if [id(node) for node in nodes] == self.node_ids:
            return

        for patch in self.node_patches:
            patch.remove()

        self.node_patches = []
        self.node_ids = [id(node) for node in nodes]

        color_map = plt.get_cmap("tab20")

        for node_id, node in enumerate(nodes):
            patch = Polygon(node.vertices, closed=True, facecolor=color_map(node_id % color_map.N), edgecolor="blue", alpha=0.16, linewidth=1.5, zorder=2)
            self.ax_main.add_patch(patch)
            self.node_patches.append(patch)

    def update_edt(self, edt_map: np.ndarray, robot_pose: tuple[float, float, float], trajectories=None) -> None:
        if self.ax_edt is None:
            return

        image = Visualizer.visualize_edt(self.ax_edt, edt_map, robot_pose, self.robot_radius, self.sensor_radius, trajectories)

        if self.edt_colorbar is None:
            self.edt_colorbar = self.fig.colorbar(image, ax=self.ax_edt, label="Obstacle clearance", fraction=0.046, pad=0.04)
            self.fig.tight_layout()
        else:
            self.edt_colorbar.update_normal(image)

    def update_controller_edt(self, edt: dict | None) -> None:
        """Draw the EDT window the controller computed and used (UDPConnection.assemble_edt), like the ROS motion planner.

        Car (rectangular footprint): the plain distance to obstacles with the rectangle on top, the gradient drawn at
        the outline point closest to an obstacle (the normal the controller uses).
        Circular robot: the clearance of the disc (distance - radius, < 0 = collision), the gradient at the robot."""
        if self.ax_edt is None or edt is None:
            return

        distances = edt["distances"]
        size_x, size_y = distances.shape
        origin_x, origin_y = edt["origin"]
        rectangle = edt["footprint"] is not None
        radius = 0.0 if rectangle else edt["robot_radius"]
        clearance = distances - radius

        # cell [i, j] is centred at origin + (i, j), one cell per map unit
        extent = (origin_x - 0.5, origin_x + size_x - 0.5, origin_y - 0.5, origin_y + size_y - 0.5)
        grad_x, grad_y = np.gradient(distances)

        if rectangle:
            i = min(max(int(round(edt["closest"][0] - origin_x)), 0), size_x - 1)
            j = min(max(int(round(edt["closest"][1] - origin_y)), 0), size_y - 1)
            gradient_origin = edt["closest"]
            label = f"footprint clearance = {edt['footprint_clearance']:.2f}"
        else:
            # the window is centred on the robot's cell
            i, j = size_x // 2, size_y // 2
            gradient_origin = None
            label = f"clearance = {clearance[i, j]:.2f}"

        image = Visualizer.draw_edt(
            ax=self.ax_edt,
            clearance=clearance,
            extent=extent,
            clim=(-radius, edt["max_distance"] - radius),
            robot_pose=edt["robot_pose"],
            robot_radius=radius,
            gradient=(grad_x[i, j], grad_y[i, j]),
            footprint=edt["footprint"],
            gradient_origin=gradient_origin,
        )
        self.ax_edt.text(0.02, 0.02, label, transform=self.ax_edt.transAxes, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})

        if self.edt_colorbar is None:
            colorbar_label = "Distance to obstacle" if rectangle else "Obstacle clearance"
            self.edt_colorbar = self.fig.colorbar(image, ax=self.ax_edt, label=colorbar_label, fraction=0.046, pad=0.04)
            self.fig.tight_layout()
        else:
            self.edt_colorbar.update_normal(image)

    def save_plots(self, run_name: str, directory: str = "results", dpi: int = 300) -> None:
        """Save each enabled panel of the figure as its own image, named <run_name>_<panel>.png."""
        edt_axes = [self.ax_edt]
        if self.edt_colorbar is not None:
            edt_axes.append(self.edt_colorbar.ax)

        panels = [
            (self.save_global_map, "global_map", [self.ax_main]),
            (self.save_local_map, "local_map", [self.ax_local]),
            (self.save_edt_map, "edt_map", edt_axes),
        ]

        if not any(enabled for enabled, _, _ in panels):
            return

        os.makedirs(directory, exist_ok=True)

        # draw once so the cropping below uses the latest layout
        self.fig.canvas.draw()

        for enabled, name, axes in panels:
            if not enabled or axes[0] is None:
                continue

            # crop the figure to this panel: the axes with its labels, title, legend (and colorbar for the EDT)
            bbox = Bbox.union([axis.get_tightbbox() for axis in axes])
            bbox = bbox.transformed(self.fig.dpi_scale_trans.inverted()).padded(0.1)

            path = os.path.join(directory, f"{run_name}_{name}.png")
            self.fig.savefig(path, bbox_inches=bbox, dpi=dpi)
            print(f"[Py] saved {path}", flush=True)

class Visualizer:
    
    @staticmethod
    def visualize_graph( obstacle_grid: np.ndarray, nodes = None, temp_goals: None = None, goal: tuple[float, float] | None = None, start: tuple[float, float] | None = None, robot_pose: tuple[float, float, float] | None = None, robot_radius: float = 2.5, sensor_radius: float = 12.0, block: bool = True, title: str | None = None, save_global_map: bool = False, save_local_map: bool = False, save_edt_map: bool = False, save_name: str = "map", save_directory: str = "results", footprint: tuple[float, float, float] | None = None) -> tuple[Figure, Axes, RobotVisualization | None]:
        # footprint: the car's box (front, rear, half width) along the heading, drawn instead of the disc of robot_radius
        # with robot_pose the save_* flags pick which panels robot_viz.save_plots() writes when the run ends;
        # without it there is only the map, and save_global_map saves it right away as <save_directory>/<save_name>_global_map.png
        columns, rows = obstacle_grid.shape

        if robot_pose is not None:
            fig, (ax, ax_local, ax_edt) = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={"width_ratios": [2.4, 1.0, 1.4]})
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax_local = None
            ax_edt = None

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

        color_map = plt.get_cmap("tab20")

        # kept so the robot visualization can replace these rectangles with update_nodes
        node_patches = []

        if nodes is not None:
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
                node_patches.append(rectangle_polygon)

        # kept so the robot visualization can move these dots instead of drawing a second set
        temp_goal_line = None

        if temp_goals:
            temp_goal_x = [cell[0] for cell in temp_goals]
            temp_goal_y = [cell[1] for cell in temp_goals]

            (temp_goal_line,) = ax.plot(
                temp_goal_x,
                temp_goal_y,
                marker="o",
                markersize=6,
                color="blue",
                linestyle="none",
                label="Temp goal",
                zorder=5,
            )

        if start is not None:
            start_x, start_y = start

            ax.plot(
                start_x,
                start_y,
                marker="o",
                markersize=9,
                color="red",
                linestyle="none",
                label="Start",
                zorder=5,
            )

        if goal is not None:
            goal_x, goal_y = goal

            ax.plot(
                goal_x,
                goal_y,
                marker="x",
                markersize=11,
                markeredgewidth=3,
                color="green",
                linestyle="none",
                label="Goal",
                zorder=5,
            )

        robot_visualization = None

        if robot_pose is not None:
            robot_visualization = RobotVisualization(fig=fig, ax_main=ax, ax_local=ax_local, obstacle_grid=obstacle_grid, robot_pose=robot_pose, robot_radius=robot_radius, sensor_radius=sensor_radius, ax_edt=ax_edt, save_global_map=save_global_map, save_local_map=save_local_map, save_edt_map=save_edt_map, footprint=footprint)

            # the startup dots become the ones update_temp_goals moves, so a replan replaces them
            robot_visualization.temp_goal_points = temp_goal_line
            robot_visualization.node_patches = node_patches

        ax.set_xlim(-0.5, columns - 0.5)

        ax.set_ylim(-0.5, rows - 0.5)

        ax.set_aspect("equal")
        ax.set_xlabel("Column, x")
        ax.set_ylabel("Row, y")

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

            legend_items.append(
                plt.Line2D(
                    [],
                    [],
                    color="blue",
                    linewidth=2.0,
                    alpha=0.8,
                    label="Robot path",
                )
            )

        # ax.legend(
        #     handles=legend_items,
        #     loc="upper right",
        # )

        plt.tight_layout()

        # map-only figure: save now, since there is no run end to save at (and show() may block)
        if robot_pose is None and save_global_map:
            os.makedirs(save_directory, exist_ok=True)
            path = os.path.join(save_directory, f"{save_name}_global_map.png")
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"[Py] saved {path}", flush=True)

        plt.show(block=block)

        if not block:
            # Run the event loop once so the window is drawn right away.
            plt.pause(0.001)

        return fig, ax, robot_visualization

    @staticmethod
    def draw_edt(ax: Axes, clearance: np.ndarray, extent: tuple[float, float, float, float], clim: tuple[float, float], robot_pose: tuple[float, float, float], robot_radius: float, gradient: tuple[float, float], footprint: tuple[float, float, float] | None = None, gradient_origin: tuple[float, float] | None = None) -> AxesImage:
        """Draw the controller's distance window, clearance indexed [x, y] (< 0 = footprint in collision), in map cells.

        footprint (front, rear, half_width), measured along the heading from the robot pose, draws a rectangle instead of
        the circle of robot_radius; the gradient arrow starts at gradient_origin (default: the robot).
        Same as Visualizer.draw_edt of the ROS motion planner (ddmr_ws)."""
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

        robot_x, robot_y, robot_theta = robot_pose[:3]
        ax.plot(robot_x, robot_y, "o", color="red", markersize=6, label="Robot")

        if footprint is None:
            ax.add_patch(Circle((robot_x, robot_y), robot_radius, fill=False, edgecolor="red", linewidth=2))
            arrow_size = robot_radius
        else:
            ax.add_patch(Polygon(Visualizer.footprint_outline((robot_x, robot_y, robot_theta), footprint), closed=True, fill=False, edgecolor="red", linewidth=2))
            arrow_size = footprint[2]

        Visualizer._draw_heading_arrow(ax=ax, robot_pose=(robot_x, robot_y, robot_theta), robot_radius=arrow_size)

        # the gradient points away from the nearest obstacle; a unit gradient is drawn a tenth of the window long
        origin_x, origin_y = gradient_origin if gradient_origin is not None else (robot_x, robot_y)
        arrow_scale = 10.0 / (extent[1] - extent[0])
        ax.quiver(origin_x, origin_y, gradient[0], gradient[1], angles="xy", scale_units="xy", scale=arrow_scale, color="magenta", width=0.006, zorder=8, label="Clearance gradient")

        ax.legend(loc="upper right")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Map x")
        ax.set_ylabel("Map y")
        ax.set_title("Controller EDT")

        return image

    @staticmethod
    def visualize_edt(ax: Axes, edt_map: np.ndarray, robot_pose: tuple[float, float, float], robot_radius: float, sensor_radius: float, trajectories=None) -> AxesImage:
        """Draw the local EDT window around the robot in map coordinates."""
        # Distance from the robot's edge, not its centre: < 0 means collision.
        clearance = edt_map - robot_radius

        robot_x, robot_y, _ = robot_pose[:3]

        # The window is centred on the robot's rounded cell, R cells to each
        # side. Cell i covers [i - 0.5, i + 0.5], same as the main map.
        radius_cells = int(np.ceil(sensor_radius))
        center_x = int(round(robot_x))
        center_y = int(round(robot_y))
        x_min = center_x - radius_cells - 0.5
        y_min = center_y - radius_cells - 0.5
        columns, rows = clearance.shape
        x_max = x_min + columns
        y_max = y_min + rows

        ax.clear()
        # edt_map is indexed [x, y]; Matplotlib images use [row, column].
        image = ax.imshow(
            clearance.T,
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            interpolation="nearest",
            aspect="equal",
        )

        # Zero clearance is where the circular footprint touches an obstacle.
        if min(clearance.shape) > 1 and clearance.min() < 0.0 < clearance.max():
            x_coords = x_min + np.arange(columns) + 0.5
            y_coords = y_min + np.arange(rows) + 0.5
            ax.contour(x_coords, y_coords, clearance.T, levels=[0.0], colors="white", linewidths=1.5)

        if trajectories is not None:
            for trajectory in trajectories:
                trajectory = np.asarray(trajectory)
                ax.plot(trajectory[:, 0], trajectory[:, 1], color="cyan", linewidth=0.8, alpha=0.5)

        ax.plot(robot_x, robot_y, "o", color="red", markersize=6, label="Robot")
        ax.add_patch(Circle((robot_x, robot_y), robot_radius, fill=False, edgecolor="red", linewidth=2))
        Visualizer._draw_heading_arrow(ax=ax, robot_pose=robot_pose, robot_radius=robot_radius)

        # The gradient points away from the nearest obstacle. The robot's
        # cell is the window centre, [radius_cells, radius_cells].
        grad_x, grad_y = np.gradient(edt_map)
        center = radius_cells
        ax.quiver(robot_x, robot_y, grad_x[center, center], grad_y[center, center], angles="xy", scale_units="xy", scale=0.5, color="magenta", width=0.006, zorder=8, label="Clearance gradient")
        ax.text(
            0.02, 0.02, f"clearance={clearance[center, center]:.3f}",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.legend(loc="upper right")

        ax.set_xlabel("Map x")
        ax.set_ylabel("Map y")
        ax.set_title("Euclidean Distance Field")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(False)
        return image

    @staticmethod
    def footprint_outline(robot_pose: tuple[float, float, float], footprint: tuple[float, float, float]) -> list[tuple[float, float]]:
        """Corners of the car's box: footprint (front, rear, half_width) measured along the heading from robot_pose."""
        robot_x, robot_y, robot_theta = robot_pose[:3]
        front, rear, half_width = footprint
        cos_theta, sin_theta = np.cos(robot_theta), np.sin(robot_theta)
        corners = [(front, half_width), (-rear, half_width), (-rear, -half_width), (front, -half_width)]
        return [(robot_x + cos_theta * u - sin_theta * v, robot_y + sin_theta * u + cos_theta * v) for u, v in corners]

    @staticmethod
    def draw_robot(ax: Axes, robot_pose: tuple[float, float, float], robot_radius: float, footprint: tuple[float, float, float] | None = None) -> tuple[Circle | Polygon, FancyArrow]:
        robot_x, robot_y, robot_theta = robot_pose[:3]

        # car: its box (front, rear, half width) along the heading; otherwise the disc of robot_radius
        if footprint is None:
            robot_patch = Circle((robot_x, robot_y), robot_radius, color="blue", alpha=0.6, zorder=6)
            arrow_size = robot_radius
        else:
            robot_patch = Polygon(Visualizer.footprint_outline((robot_x, robot_y, robot_theta), footprint), closed=True, color="blue", alpha=0.6, zorder=6)
            arrow_size = footprint[2]

        ax.add_patch(robot_patch)

        robot_arrow = Visualizer._draw_heading_arrow(ax=ax, robot_pose=(robot_x, robot_y, robot_theta), robot_radius=arrow_size)

        return robot_patch, robot_arrow

    @staticmethod
    def extract_local_map(obstacle_grid: np.ndarray, robot_pose: tuple[float, float, float], sensor_radius: float, visible_mask: np.ndarray | None = None) -> np.ndarray:
        columns, rows = obstacle_grid.shape

        robot_x, robot_y, _ = robot_pose[:3]

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

        distance_squared = (
            (local_rows + center_row - radius_cells - robot_y) ** 2
            + (local_columns + center_column - radius_cells - robot_x) ** 2
        )

        local_map[distance_squared > sensor_radius**2] = 0.5

        return local_map

    @staticmethod
    def extract_local_occupancy(obstacle_grid: np.ndarray, robot_pose: tuple[float, float, float], sensor_radius: float, visible_mask: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        if visible_mask is None:
            visible_mask = Visualizer.visibility_mask(obstacle_grid, robot_pose, sensor_radius)

        local_map = Visualizer.extract_local_map(obstacle_grid, robot_pose, sensor_radius, visible_mask)
  
        return local_map.T == 1.0

    @staticmethod
    def local_free_points(obstacle_grid: np.ndarray, robot_pose: tuple[float, float, float], sensor_radius: float, stride: int = 3, visible_mask: np.ndarray | None = None) -> np.ndarray:
        columns, rows = obstacle_grid.shape

        robot_x, robot_y, _ = robot_pose[:3]

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
        """Find visible global cells; pose, sensor radius, and ray step use cells."""
        columns, rows = obstacle_grid.shape

        robot_x, robot_y, _ = robot_pose[:3]

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
    def _draw_heading_arrow(ax: Axes, robot_pose: tuple[float, float, float], robot_radius: float, color: str = "darkblue") -> FancyArrow:
        robot_x, robot_y, robot_theta = robot_pose[:3]

        arrow_length = robot_radius * 1.5

        return ax.arrow(
            robot_x,
            robot_y,
            arrow_length * np.cos(robot_theta),
            arrow_length * np.sin(robot_theta),
            head_width=robot_radius * 0.6,
            head_length=robot_radius * 0.4,
            fc=color,
            ec=color,
            linewidth=2,
            zorder=7,
        )
