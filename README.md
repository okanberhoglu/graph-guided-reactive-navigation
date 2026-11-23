# graph-guided-reactive-navigation
Graph-guided reactive navigation for nonholonomic ground vehicles using GPU-accelerated sampling-based MPC.

## Usage

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

Only `robot` mode takes a map. Select one with `-m`/`--map`, followed by a map number:

```bash
python main.py robot --map 4
python main.py robot -m 4        # short form
```

Maps `1` through `6` are available. Omitting the flag loads map `1`.
