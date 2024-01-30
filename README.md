# Simple Density Matrix Renormalization Group Package - simple_dmrg

`simple_dmrg` is a Python package of a simple implementation of DMRG, based on the pseudocode in Fig. 19 of Catarina and Murta:  https://doi.org/10.1140/epjb/s10051-023-00575-2, along with drawing on parts of Hubig et al. https://link.aps.org/doi/10.1103/PhysRevB.95.035129 and Dolfi et al. http://dx.doi.org/10.1016/j.cpc.2014.08.019.

# Installation

## Environment Setup and Package Installation

We recommend setting up and running this code on a Linux flavour OS, MacOSX, or Windows Subsystem for Linux 2 (WSL2). It has been tested so far on WSL2. If you have any issues, please [contact us](#contact-information).


### Python Installation
First, `Python 3.9` is needed. We recommend using the Python version manager `pyenv`  to install this ([instructions are here](https://realpython.com/intro-to-pyenv/)). `pyenv` is under the open source MIT-license ([Github Repository](https://github.com/pyenv/pyenv)). We specify specifically `Python 3.9` for the sake of consistency in the benchmark.

### Obtain the source code
To obtain the source code, clone the git repository or download the code directly. Then, navigate to the code location (i.e. where pyproject.toml is located; this directory will be called `simple_dmrg` if the code is cloned).

### Virtual Environment Setup
Once `Python 3.9` is installed and activated in the terminal, we recommend creating a new virtual environment in the code folder using `venv` and then installing  the package via `pip`.

More detailed instructions to create a Python virtual environment can be found [here](https://realpython.com/python-virtual-environments-a-primer/), with our suggested commands below.

#### Create the environment
First, navigate to the code folder and run:
```
python -m venv dmrg_env
```
#### Activate the environment
Before installing, the environment must be activated. For example, by running
```
source dmrg_env/bin/activate
```
Whenever you wish to deactivate the environment, run
```
deactivate
```

### Prerequisites
Any needed dependencies should be installed automatically.

### Installation
To install `simple_dmrg`, first ensure the correct virtual environment is activated.
Then, in the code directory (i.e. where pyproject.toml is located; this directory will be called `simple_dmrg` if the code is cloned) run:
```
python -m pip install .
```
[//]: # (## Note: simple_dmrg will likely be posted on PyPi in the future to make everything easier.)

# Examples
See the examples directory for a python notebook showing how to run the code. There is also an html version for easy viewing.

# Contact Information

We can be reached at joshua.cantin@utoronto.ca 