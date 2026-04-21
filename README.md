[![Documentation Status](https://readthedocs.org/projects/pyneurode/badge/?version=latest)](https://pyneurode.readthedocs.io/en/latest/?badge=latest)

# pyneurode
pyneurode (Python + neuron + node) is a real-time signal processing framework for neural recordings, with a special focus on online spike sorting and neural signal decoding in the Open Ephys GUI. It has a robust architecture for parallel processing and can sort hundreds of channel of tetrode signals in real-time.

<img width="2721" height="1483" alt="screenshot2" src="https://github.com/user-attachments/assets/0cb2bb78-e842-4e79-bc06-d2d63363f40c" />

For technical details please see our [preprint](https://www.biorxiv.org/content/10.1101/2022.01.18.476764v1)

Please see the [documentation](https://pyneurode.readthedocs.io/en/latest/) for how to use the library.

The project is still under heavy development. Please submit an issue if you encounter any problem.


## Installation
1. Install uv for your platform [link](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone this repository
3. Under project root, run `uv sync`

Note: you will need to have build tools (MSVC, clang etc) to build the `isosplit` package required by pyneurode.

## Usage

Consult the examples workflow in the `example/workflows` folder. 
To run the workflow, e.g.
```
uv run examples\workflows\sorter_node_openephys_replay.py
```
