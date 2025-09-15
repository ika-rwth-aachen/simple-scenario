from pathlib import Path
from simple_scenario import Scenario


def main():
    scenario_dir = Path(__file__).parent

    for i in range(4):
        scenario_file = scenario_dir / f"example_scenario{i + 1}.json"
        scenario = Scenario.from_x(scenario_file)
        scenario.render_gif(scenario_dir)


if __name__ == "__main__":
    main()
