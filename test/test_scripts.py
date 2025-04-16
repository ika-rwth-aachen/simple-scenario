from pathlib import Path

from scripts import create_scenario


class TestScripts:
    DATA_DIR = Path(__file__).parent / ".." / "scripts"

    def test_scripts_create_scenario(self):
        scenario_config = self.DATA_DIR / "simple_scenario.json"

        result_dir = create_scenario.main(
            scenario_config,
        )

        # Check that all files have been created
        assert (result_dir / f"{scenario_config.stem}.xosc").is_file()
        assert (result_dir / f"{scenario_config.stem}.xodr").is_file()
        assert (result_dir / f"{scenario_config.stem}.osm").is_file()
        assert (result_dir / f"{scenario_config.stem}.png").is_file()
