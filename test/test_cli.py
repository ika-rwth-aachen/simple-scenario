from pathlib import Path

from simple_scenario.cli import create


class TestCLI:
    DATA_DIR = Path(__file__).parent / ".." / "demo"

    RESULT_DIR = Path(__file__).parent / "results" / "test_cli"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_cli_demo_default(self):
        result_dir = self.RESULT_DIR / "test_cli_demo_default"
        result_dir.mkdir(exist_ok=True)

        scenario_config = self.DATA_DIR / "cutout_example.json"

        create(
            scenario_config,
            result_dir,
        )

        # Check that all files have been created
        openx_dir = result_dir / f"{scenario_config.stem}_openx"
        assert openx_dir.is_dir()
        assert (openx_dir / f"{scenario_config.stem}.xosc").is_file()
        assert (openx_dir / f"{scenario_config.stem}.xodr").is_file()

    def test_cli_demo_with_lanelet2(self):
        result_dir = self.RESULT_DIR / "test_cli_demo_with_lanelet2"
        result_dir.mkdir(exist_ok=True)

        scenario_config = self.DATA_DIR / "cutout_example.json"

        create(
            scenario_config,
            result_dir,
            lanelet2=True,
        )

        # Check that all files have been created
        openx_dir = result_dir / f"{scenario_config.stem}_openx"
        assert openx_dir.is_dir()
        assert (openx_dir / f"{scenario_config.stem}.xosc").is_file()
        assert (openx_dir / f"{scenario_config.stem}.xodr").is_file()
        assert (openx_dir / f"{scenario_config.stem}.osm").is_file()
