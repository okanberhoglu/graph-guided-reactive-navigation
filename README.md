# graph-guided-reactive-navigation
Graph-guided reactive navigation for nonholonomic ground vehicles using GPU-accelerated sampling-based MPC.

The repository is a set of ROS 2 packages. A global planner covers the free space of a static map with rectangular regions, finds a route through them to the goal, and hands the controller one waypoint at a time together with the regions leading to it. The controller then drives reactively against a local occupancy grid built from the laser scan.

## Packages

| Package | Type | Contents |
|---|---|---|
| `global_motion_planner` | ament_python | `motion_planner_node`: region graph, Dijkstra route, waypoint supervision, Matplotlib views |
| `local_map_creator` | ament_python | `map_creator_node`: rolling occupancy grid around the vehicle from `LaserScan` |
| `messages` | ament_cmake | Custom messages: `RectRegion`, `RectRegionArray`, `EdtWindow` |
| `robot_controller` | ament_cmake | `controller_node`: GPU sampling-based MPC for a differential-drive robot (CUDA) |
| `car_controller` | ament_cmake | `controller_node`: GPU sampling-based MPC for a car-like (Ackermann) vehicle, tuned for the CARLA Lincoln MKZ (CUDA) |
| `graph_guided_nav_bringup` | ament_cmake | Launch files and RViz configs for the robot and the CARLA car |


## Requirements

- Ubuntu 22.04 with ROS 2 Humble
- Python 3.10 with NumPy, SciPy and Matplotlib
- An NVIDIA GPU with the CUDA toolkit 11.5 or newer (`nvcc`) for the controllers
- A display, since the planner opens Matplotlib windows
- Optional: [CARLA](https://carla.org/) with the CARLA ROS bridge

## Build

1. Create a workspace with a `src` folder:

   ```bash
   mkdir -p ~/graph_guided_nav_ws/src
   ```

2. Clone the `ros2-humble` branch into `src`. The default branch, `main`, holds the earlier stand-alone Python version without the ROS 2 packages, so the `-b ros2-humble` is needed:

   ```bash
   cd ~/graph_guided_nav_ws/src
   git clone -b ros2-humble https://github.com/okanberhoglu/graph-guided-reactive-navigation.git
   ```

3. Install the dependencies declared in each `package.xml`:

   ```bash
   cd ~/graph_guided_nav_ws
   source /opt/ros/humble/setup.bash
   rosdep install --from-paths src --ignore-src -y
   ```

   If `rosdep` is not set up yet, run `sudo apt install python3-rosdep && sudo rosdep init && rosdep update` once.

4. Build from the workspace root, not from inside the repository:

   ```bash
   colcon build --symlink-install
   ```

5. Source the workspace. Do this in every new terminal before running the nodes or launch files:

   ```bash
   source ~/graph_guided_nav_ws/install/setup.bash
   ```

## Run

### With the launch files

`graph_guided_nav_bringup` starts the map creator, the planner, the matching controller and RViz with one command. Source `install/setup.bash` first.

Differential-drive robot:

```bash
ros2 launch graph_guided_nav_bringup robot.launch.py
```

CARLA car:

```bash
ros2 launch graph_guided_nav_bringup car.launch.py
```

| Launch file | `is_car` | Controller | RViz config |
|---|---|---|---|
| `robot.launch.py` | `false` | `robot_controller` | `rviz/robot.rviz` |
| `car.launch.py` | `true` | `car_controller` | `rviz/car.rviz` |

Both launch files take the same arguments:

| Argument | Default | Meaning |
|---|---|---|
| `use_sim_time` | `false` | Run every node and RViz on `/clock` instead of wall time. Set it to `true` only when a simulator publishes `/clock`: Gazebo does, the CARLA ROS 2 bridge used here does not. With `true` and no `/clock`, the nodes never tick and `/local_map` stays empty. |
| `rviz` | `true` | Start RViz |

```bash
ros2 launch graph_guided_nav_bringup robot.launch.py use_sim_time:=true   # robot in Gazebo
ros2 launch graph_guided_nav_bringup car.launch.py rviz:=false            # car without RViz
```

The launch files start only this repository's nodes. The map on `/map`, the laser scan, the odometry and the TF tree have to come from elsewhere; see [What the nodes wait for](#what-the-nodes-wait-for). For example, a TurtleBot3 Burger in Gazebo with Cartographer providing the map:

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
ros2 launch graph_guided_nav_bringup robot.launch.py use_sim_time:=true
```

The planner builds its region graph once, from the first map it receives, so finish mapping before starting the launch file, or restart it once the map is complete.

For the car, start CARLA and the CARLA ROS 2 bridge first, then `car.launch.py`.

**RViz.** Both configs show the global map (drawn behind everything), the local map, the temporary goal, the final goal, the laser scan, the odometry and TF. The car config also shows the ego bounding box (`/carla/ego/bounding_box`) and the CARLA path (`/carla/ego/path`). The view is a 3D orbit camera that follows `base_link`; switch to the top-down view under *Saved* in the Views panel. The **2D Goal Pose** tool sends the goal to the planner on `/goal_pose`. With `--symlink-install`, *File → Save Config* writes your changes back to the package's `.rviz` file.

### Node by node

To run the nodes without the launch files, start each in its own terminal after sourcing `install/setup.bash`. Run only one controller at a time: both use the node name `reactive_controller`.

Differential-drive robot:

```bash
ros2 run local_map_creator map_creator_node
ros2 run global_motion_planner motion_planner_node
ros2 run robot_controller controller_node
```

CARLA car:

```bash
ros2 run local_map_creator map_creator_node --ros-args -p is_car:=true
ros2 run global_motion_planner motion_planner_node --ros-args -p is_car:=true
ros2 run car_controller controller_node
```

Add `--ros-args -p use_sim_time:=true` to each node when the simulator publishes `/clock`.

### What the nodes wait for

All nodes start idle and wait for their inputs. The topic names below are the robot defaults; see [Car mode](#car-mode-is_car) for the car ones.

**`map_creator_node`** publishes `/local_map` once it has:

- A `sensor_msgs/LaserScan` on `/scan`
- TF from the scan's frame to `map` and from `base_link` to `map`, at the scan's timestamp. Scans without both transforms are skipped.

**`motion_planner_node`** needs these, in this order:

1. **A map on `/map`** (`nav_msgs/OccupancyGrid`). The planner subscribes with reliable, transient-local QoS, so the map publisher must use the same; a volatile publisher is incompatible and the map never arrives. The region graph is built once, from the first map.
2. **Odometry on `/odom`** (`nav_msgs/Odometry`), plus TF from the odometry message's frame to the map's frame. Odometry received before the map is ignored.
3. **A goal on `/goal_pose`** (`geometry_msgs/PoseStamped`). A goal that arrives before the map or the first pose is dropped, not queued, so send it again.

Once it has all three, it plans a route and starts publishing `/temp_goal_point` and `/current_nodes`.

**The controllers** (`controller_node` in `robot_controller` and in `car_controller`) publish a drive command every 0.1 s once they have:

- Odometry on `/odom` (car: `/carla/ego/odometry`), plus TF from the odometry message's frame to `map`. The TF is not needed when the odometry is already in `map`.
- A temporary goal on `/temp_goal_point`, from the planner
- A local map on `/local_map`, from the map creator

`/temp_goal_point`, `/current_nodes` and `/local_map` must be in the `map` frame; messages in any other frame are ignored with a warning. `/current_nodes` is optional: while it is empty, the controller does not keep the vehicle inside the route regions.

Each controller stops once it is within `goalTolerance` (`params.h`) of its goal, then waits for a different goal:

- **Robot:** distance from the robot centre. It sends one zero `Twist`.
- **Car:** distance from the look-ahead point on the front axle. It sends one command with zero speed and steering, braking at `maxDecel`.

The robot controller ticks on wall time. The car controller ticks on the node clock, so with `use_sim_time:=true` it follows the simulator's `/clock` and does nothing until `/clock` is published.

### Sending a goal

Send the goal with the **2D Goal Pose** tool in RViz 2, or from the command line. The planner reads the goal position as `map` coordinates and ignores the orientation:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 10.0, y: 5.0}}}"
```

The planner shows the occupancy grid with the graph, route and temporary goals in one window. Once a controller publishes `/controller_edt`, a second window shows the controller's obstacle distance field.

### Car mode (`is_car`)

`map_creator_node` and `motion_planner_node` both take a boolean `is_car` parameter, `false` by default. `car.launch.py` sets it to `true` on both; when running node by node, pass it as in the car commands above. The parameter is read once at startup, so restart the node to change it.

| | Robot (`is_car:=false`, default) | Car (`is_car:=true`) |
|---|---|---|
| Laser scan | `/scan` | `/carla/ego/scan` |
| Odometry | `/odom` | `/carla/ego/odometry` |
| Local map (`/local_map`) | 8 m × 8 m | 20 m × 20 m |
| Region distance from obstacles (`distance_threshold`) | 0.1 m | 1.0 m |
| Extra clearance a new region needs (`min_rect_half_size`) | 0.1 m | 0.5 m |
| Waypoint reach radius (`goal_distance_threshold`) | 0.5 m | 4.0 m |

The controllers have no `is_car` parameter; choose the package instead: `robot_controller` with `is_car:=false`, `car_controller` with `is_car:=true`.

## Topics

**`map_creator_node`**

| Direction | Topic | Type | Notes |
|---|---|---|---|
| Subscribes | `/scan` (car: `/carla/ego/scan`) | `sensor_msgs/LaserScan` | Sensor-data QoS |
| Publishes | `/local_map` | `nav_msgs/OccupancyGrid` | Grid in the `map` frame, centred on `base_link` and rebuilt from each scan, 0.1 m cells |

**`motion_planner_node`**

| Direction | Topic | Type | Notes |
|---|---|---|---|
| Subscribes | `/map` | `nav_msgs/OccupancyGrid` | Transient local; the first message builds the graph |
| Subscribes | `/odom` (car: `/carla/ego/odometry`) | `nav_msgs/Odometry` | Vehicle pose, transformed into the map frame |
| Subscribes | `/goal_pose` | `geometry_msgs/PoseStamped` | Final goal |
| Subscribes | `/controller_edt` | `messages/EdtWindow` | Optional; the controller's distance field, only displayed |
| Publishes | `/temp_goal_point` | `geometry_msgs/PoseStamped` | Current temporary goal, heading toward the next one |
| Publishes | `/current_nodes` | `messages/RectRegionArray` | Regions from the vehicle to the temporary goal, in metres, every 0.1 s |

**Controllers**: `controller_node` in `robot_controller` and in `car_controller`, both with the node name `reactive_controller`

| Direction | Topic | Type | Notes |
|---|---|---|---|
| Subscribes | `/odom` (car: `/carla/ego/odometry`) | `nav_msgs/Odometry` | Vehicle pose, transformed into `map` with TF; the car also takes its speed from the twist |
| Subscribes | `/temp_goal_point` | `geometry_msgs/PoseStamped` | Goal to drive to, `map` frame |
| Subscribes | `/current_nodes` | `messages/RectRegionArray` | Regions to stay inside, `map` frame; optional |
| Subscribes | `/local_map` | `nav_msgs/OccupancyGrid` | Obstacles, `map` frame |
| Publishes (robot) | `/cmd_vel` | `geometry_msgs/Twist` | `linear.x` and `angular.z`, every 0.1 s |
| Publishes (car) | `/carla/ego/ackermann_cmd` | `ackermann_msgs/AckermannDriveStamped` | `speed`, `steering_angle` and `acceleration`, every 0.1 s; stamped in the odometry's `child_frame_id` (`base_footprint` if empty) |
| Publishes | `/controller_edt` | `messages/EdtWindow` | Distance field used on the last tick; only built while something subscribes |

Parameters, set with `--ros-args -p <name>:=<value>`:

| Parameter | Robot default | Car default | Meaning |
|---|---|---|---|
| `map_frame` | `map` | `map` | Frame the goal, regions and local map must be in |
| `cell_size` | `0.1` | `0.1` | Metres per controller cell. The tuning in `params.h` is in cells, so changing this rescales all of it |
| `sensor_radius_cells` | `20.0` (2 m) | `90.0` (9 m) | Radius of the distance window around the vehicle, in cells |

The remaining tuning (horizon, sample count, speed and steering limits, cost weights, `goalTolerance`) is compiled in from `params.h` and needs a rebuild: [robot](robot_controller/include/robot_controller/params.h), [car](car_controller/include/car_controller/params.h) (Lincoln MKZ 2020 in CARLA, kinematic bicycle model).

## License

MIT. See [LICENSE](LICENSE).
