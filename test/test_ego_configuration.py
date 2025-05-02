from pathlib import Path

from simple_scenario import EgoConfiguration
from simple_scenario.rendering import create_scenario_plot_ax
from simple_scenario.road import Road, StraightSegment


class TestEgoConfiguration:
    RESULT_DIR = Path(__file__).parent / "results" / "test_ego_configuration"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_ego_configuration(self):
        result_dir = self.RESULT_DIR / "test_ego_configuration"
        result_dir.mkdir(exist_ok=True)

        road = Road(3, 3.75, [StraightSegment(200)])

        ego_configuration = EgoConfiguration(1000, 50, 0, 27.78)

        ego_configuration.compile(road)

        ego_configuration.render(result_dir, "ego_configuration")
        ego_configuration.render(result_dir, "ego_configuration_clean", clean=True)

        with create_scenario_plot_ax(result_dir, "ego_configration_on_road") as ax:
            road.render(ax)
            ego_configuration.render(ax)

        with create_scenario_plot_ax(
            result_dir, "ego_configration_on_road_clean", clean=True
        ) as ax:
            road.render(ax)
            ego_configuration.render(ax)


if __name__ == "__main__":
    tester = TestEgoConfiguration()
    tester.test_ego_configuration()
