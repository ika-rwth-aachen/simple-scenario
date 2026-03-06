import pytest

from pathlib import Path

from simple_scenario import (
    Scenario,
)


class TestScenarioOnMappedRoad:
    DATA_DIR = Path(__file__).parent / "assets" / "synthetic-curve"
    RESULT_DIR = Path(__file__).parent / "results" / "test_scenario_on_mapped_road"
    RESULT_DIR.mkdir(exist_ok=True)

    @staticmethod
    def _create_result_dir(test_name: str) -> Path:
        result_dir = TestScenarioOnMappedRoad.RESULT_DIR / test_name
        result_dir.mkdir(exist_ok=True)
        return result_dir

    @staticmethod
    def _stress_test(scenario: Scenario, result_dir: str) -> Path:
        scenario.save(result_dir, mode="openx")
        scenario.save(result_dir, mode="lanelet2")

        scenario.render(result_dir)
        with pytest.raises(
            NotImplementedError, match="Cannot create gif of mapped road."
        ):
            scenario.render_gif(result_dir)

    def test_lane_change(self):
        result_dir = self._create_result_dir("test_mapped_road_lane_change")
        scenario_config = self.DATA_DIR / "lane_change.json"

        scenario = Scenario.from_x(scenario_config)

        self._stress_test(scenario, result_dir)


if __name__ == "__main__":
    tester = TestScenarioOnMappedRoad()
    tester.test_lane_change()
