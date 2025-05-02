import pytest
from pathlib import Path

from simple_scenario import CR_AVAILABLE, LXD_AVAILABLE

if LXD_AVAILABLE and CR_AVAILABLE:
    from simple_scenario.lxd import HighdExtractor


class TestSimpleScenarioExtractor:
    DATA_DIR = Path(__file__).parent / "assets/highD-dataset-v1.0"

    RESULT_DIR = Path(__file__).parent / "results" / "test_simple_scenario_extractor"
    RESULT_DIR.mkdir(exist_ok=True)

    @pytest.mark.skipif(
        not (CR_AVAILABLE and LXD_AVAILABLE),
        reason="commonroad and lxd extras are not installed",
    )
    def test_simple_scenario_extraction_highd(self):
        result_dir = self.RESULT_DIR / "test_simple_scenario_extraction_highd"
        result_dir.mkdir(exist_ok=True)

        extractor = HighdExtractor(self.DATA_DIR)

        simple_scenarios = extractor.extract_simple_scenarios()

        _, s, _ = next(simple_scenarios[1])

        s.get_cr_interface()

        s.render(result_dir, dpi=600)
        s.render_gif(result_dir, dpi=600)


if __name__ == "__main__":
    tester = TestSimpleScenarioExtractor()
    tester.test_simple_scenario_extraction_highd()
