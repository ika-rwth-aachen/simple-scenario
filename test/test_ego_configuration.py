from pathlib import Path

import numpy as np

from simple_scenario import EgoConfiguration
from simple_scenario.rendering import create_scenario_plot_ax
from simple_scenario.road import SyntheticRoad, StraightSegment


class TestEgoConfiguration:
    RESULT_DIR = Path(__file__).parent / "results" / "test_ego_configuration"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_ego_configuration(self):
        result_dir = self.RESULT_DIR / "test_ego_configuration"
        result_dir.mkdir(exist_ok=True)

        road = SyntheticRoad(3, 3.75, [StraightSegment(200)])

        ego_configuration = EgoConfiguration(
            start_lanelet_id=1000,
            start_s=50,
            start_t=0,
            v0=27.78,
            target_s=150,
            target_t=0,
        )

        ego_configuration.compile(road)

        ego_configuration.render(result_dir, "ego_configuration")
        ego_configuration.render(result_dir, "ego_configuration_clean", clean=True)

        with create_scenario_plot_ax(result_dir, "ego_configration_on_road") as ax:
            road.render(ax)
            ego_configuration.render(ax)

    def test_ego_configuration_target_xy(self):
        result_dir = self.RESULT_DIR / "test_ego_configuration_target_xy"
        result_dir.mkdir(exist_ok=True)

        road = SyntheticRoad(3, 3.75, [StraightSegment(200)])

        target_lanelet_id = 1000
        target_s = 150
        target_t = 0.0
        target_x, target_y = road.from_frenet_to_cart(
            road.lanelet_map.laneletLayer[target_lanelet_id].centerline,
            target_s,
            target_t,
        )

        ego_configuration = EgoConfiguration(
            start_lanelet_id=target_lanelet_id,
            start_s=50,
            start_t=0,
            v0=27.78,
            target_s=None,
            target_t=None,
            target_lanelet_id=None,
            target_x=target_x,
            target_y=target_y,
        )

        ego_configuration.compile(road)

        assert ego_configuration.target_lanelet_id == target_lanelet_id
        assert np.isclose(ego_configuration.target_s, target_s)
        assert np.isclose(ego_configuration.target_t, target_t)
        assert np.isclose(ego_configuration.route_length, target_s - 50)

        with create_scenario_plot_ax(
            result_dir, "ego_configration_on_road_clean", clean=True
        ) as ax:
            road.render(ax)
            ego_configuration.render(ax)


if __name__ == "__main__":
    tester = TestEgoConfiguration()
    tester.test_ego_configuration()
