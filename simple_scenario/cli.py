from __future__ import annotations

import sys
import typer

from loguru import logger
from pathlib import Path

from .scenario import Scenario

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def hello_world() -> None:
    print("hello world!")


@app.command()
def create(config_file: str, result_dir: str = None, osc: bool = True) -> None:  # noqa: RUF013
    logger.configure(handlers=[{"sink": sys.stdout, "level": "WARNING"}])

    config_file = Path(config_file)

    if not config_file.is_file():
        msg = "Please provide a valid config file."
        raise FileNotFoundError(msg)

    if result_dir is None:
        result_dir = config_file.parent
    result_dir = Path(result_dir)

    scenario = Scenario.from_config_file(config_file)

    scenario_result_dir = result_dir / f"{scenario.id}_openx"
    scenario_result_dir.mkdir(exist_ok=True)

    if osc:
        print(f"Create openX scenario at: {scenario_result_dir.resolve()}/")

        scenario.save(scenario_result_dir, mode="openx")

        xosc_file = scenario_result_dir / f"{scenario.id}.xosc"

        print(f"To run use: esmini --window --osc {xosc_file.resolve()}")


if __name__ == "__main__":
    app()
