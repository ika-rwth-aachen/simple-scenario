from pathlib import Path

from simple_scenario import Scenario


def main(config: str) -> Path:
    result_dir = Path(config).parent
    result_dir.mkdir(exist_ok=True)

    scenario = Scenario.from_x(config)

    scenario.render(result_dir)
    scenario.save(result_dir, mode="openx")
    scenario.save(result_dir, mode="lanelet2")

    return result_dir


if __name__ == "__main__":
    main("scripts/simple_scenario.json")
