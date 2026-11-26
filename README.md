# graph-guided-reactive-navigation
Graph-guided reactive navigation for nonholonomic ground vehicles using GPU-accelerated sampling-based MPC.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with the CUDA toolkit 11.5 or newer (`nvcc`) for the controllers

## Usage

### Simulation

First, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Then run a mode with `main.py`. The mode is optional and defaults to `robot`:

```bash
python main.py                  # same as: python main.py robot
python main.py <mode>
python main.py --help           # list the available modes and maps
```

Five modes are available: `robot`, `car`, `social_cross`, `social_counter`, and `global`. Pass one as the single positional argument, for example `python main.py social_cross`. Omitting it runs `robot`, and any value outside this set exits with an error listing the valid modes.

Only `robot` and `car` modes take a map. Select one with `-m`/`--map`, followed by a map number:

```bash
python main.py robot --map 4
python main.py robot -m 4        # short form
python main.py car --map 4
```

Maps `1` through `6` are available. Omitting the flag loads map `1`. In `car` mode the map is scaled up 4x.

### Robot Controller

The robot is driven by the GPU controller in `robot_controller/`, which needs the CUDA toolkit (`nvcc`). Build it once:

```bash
cd robot_controller
nvcc -std=c++17 -O2 -o controller controller.cpp controller.cu
```

Start the controller first, then run a mode in a second terminal:

```bash
./robot_controller/controller   # terminal 1
python main.py robot            # terminal 2
```

The controller shuts down when the robot reaches the goal.

### Car Controller

The car is driven by the GPU controller in `car_controller/`, which also needs the CUDA toolkit (`nvcc`). Build it once:

```bash
cd car_controller
nvcc -std=c++17 -O2 -o controller controller.cpp controller.cu
```

Start the controller first, then run `car` mode in a second terminal:

```bash
./car_controller/controller     # terminal 1
python main.py car --map 4      # terminal 2
```

The controller shuts down when the car reaches the goal or hits an obstacle.

The robot and car controllers use the same UDP port (`5001`), so run only one of them at a time.
