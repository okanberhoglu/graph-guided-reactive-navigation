# graph-guided-reactive-navigation
Graph-guided reactive navigation for nonholonomic ground vehicles using GPU-accelerated sampling-based MPC.

## Usage

Run a mode with `main.py`. The mode is optional and defaults to `robot`:

```bash
python main.py                  # same as: python main.py robot
python main.py <mode>
```

Five modes are available: `robot`, `car`, `social_cross`, `social_counter`, and `global`. Pass one as the single positional argument, for example `python main.py social_cross`. Omitting it runs `robot`, and any value outside this set exits with an error listing the valid modes.

Only `robot` mode takes a map. Select one with `-m`/`--map`, followed by a map number:

```bash
python main.py robot --map 4
python main.py robot -m 4        # short form
```

Map numbers come from the `get_map<N>` methods of `MapCreator` in `utils/utils.py`, currently `1` through `7`. Omitting the flag loads map `1`. Because the other modes do not use a map, passing `--map` to any of them is an error rather than a silently ignored flag — `python main.py car --map 3` exits with a message naming the modes that accept it. An out-of-range number such as `--map 9` exits with the list of valid numbers, and `python main.py --help` prints that same list, so adding a new `get_map8` method makes it available here without further changes.
