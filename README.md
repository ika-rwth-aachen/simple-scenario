# simple-scenario

Quickly generate prototype test scenarios for automated driving.

# Notice

> [!IMPORTANT]
> This repository is open-sourced and maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/).
> We cover a wide variety of research topics within our [*Vehicle Intelligence & Automated Driving*](https://www.ika.rwth-aachen.de/en/competences/fields-of-research/vehicle-intelligence-automated-driving.html) domain.
> If you would like to learn more about how we can support your automated driving or robotics efforts, feel free to reach out to us!
> :email: ***opensource@ika.rwth-aachen.de***

# Install

```bash
pip install simple-scenario
```

# Use

`simple-scenario` offers a rich python API and a basic CLI for test scenario generation.

## Python API

The python API supports various ways of generating and interacting with test scenarios.

Please conduct the [tutorials/A_01_getting_started.ipynb](https://github.com/ika-rwth-aachen/simple-scenario/blob/main/tutorials/A_01_getting_started.ipynb) to learn about the basic usage of `simple-scenario`.

Please find more in-depth examples in the [test/](https://github.com/ika-rwth-aachen/simple-scenario/tree/main/test) folder (E.g., [test/test_scenario.py](https://github.com/ika-rwth-aachen/simple-scenario/blob/main/test/test_scenario.py) shows which input formats can be used).

> :bulb: *Have a look at the [Dev](#dev) section of this readme to find out how to setup the project and run the tutorial notebooks and tests.*

## CLI

The CLI supports generating a set of

* an OpenSCENARIO (`.xosc`),
* an OpenDRIVE (`.xodr`),
* and a Lanelet2 (`.osm`) file

for a given `simple-scenario` config file (`.json`) through the following command

```bash
simple-scenario create demo/cutout_example.json
```

# Extras

## commonroad-extra

> :warning: *Needs python version 3.8*

```bash
pip install simple-scenario[commonroad]
```

Enables advanced features like feasibility checking of the scenario (powered by the [CommonRoad Drivability Checker](https://commonroad.in.tum.de/tools/drivability-checker)) and exporting the scenario to the [CommonRoad](https://commonroad.in.tum.de/) (CR) format.

Please conduct the [tutorials/B_01_commonroad_extra.ipynb](https://github.com/ika-rwth-aachen/simple-scenario/blob/main/tutorials/B_01_commonroad_extra.ipynb) for more details.

## lxd-extra

```bash
pip install simple-scenario[lxd]
```

Enables reading [leveLXData](https://levelxdata.com/) datasets through [lxd-io](https://github.com/lenvt/lxd-io).

> :warning: *Currently only the [highD dataset](https://highd-dataset.com) is supported*

Please conduct the [tutorials/C_01_lxd_extra.ipynb](https://github.com/ika-rwth-aachen/simple-scenario/blob/main/tutorials/C_01_lxd_extra.ipynb) for more details.

# Dev

To develop `simple-scenario`, you must first clone the repository.

```bash
$ git clone git@gitlab.ika.rwth-aachen.de:fb-fi/scenarios/simple-scenario.git
$ cd simple-scenario
```

It is recommended to use [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management. If you do not want to use `uv`, please consult the [Without uv](#without-uv) section.

## With uv

Install requirements with

```bash
$ uv sync
```

To run a script, use

```bash
$ uv run /path/to/script.py
```

or directly use the python interpreter from the `.venv` folder in e.g. VSCode.

To run the tests, install the dev requirements with

```bash
$ uv sync --dev
```

and run the tests

```bash
$ uv run pytest
```

To run the notebooks in the [tutorials/](https://github.com/ika-rwth-aachen/simple-scenario/tree/main/tutorials) folder in VSCode, select the `.venv`'s as the kernel (see [here](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code) for more info).

## Without uv

Install the project editable

```bash
$ python -m pip install -e .
```

To run the tests, first install pytest

```bash
$ python -m pip install pytest
```

and run

```bash
$ pytest
```

# Acknowledgements

This package is developed as part of the [SYNERGIES project](https://synergies-ccam.eu).

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/simple-scenario/refs/heads/main/assets/synergies.svg" style="width:2in" />

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor the granting authority can be held responsible for them.

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/simple-scenario/refs/heads/main/assets/funded_by_eu.svg" style="width:4in" />
