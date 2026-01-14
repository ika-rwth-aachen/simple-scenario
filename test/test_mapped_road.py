import pytest

from pathlib import Path

from simple_scenario.road import MappedRoad


class TestMappedRoad:
    DATA_DIR = Path(__file__).parent / "assets" / "atc"
    OPENDRIVE_MAP = DATA_DIR / "aldenhoven.xodr"
    LANELET2_MAP = DATA_DIR / "aldenhoven.osm"

    RESULT_DIR = Path(__file__).parent / "results" / "test_mapped_road"
    RESULT_DIR.mkdir(exist_ok=True)

    @staticmethod
    def _create_result_dir(test_name: str) -> Path:
        result_dir = TestMappedRoad.RESULT_DIR / test_name
        result_dir.mkdir(exist_ok=True)
        return result_dir

    @staticmethod
    def _stress_test(test_name: str, road: MappedRoad, result_dir: Path) -> None:
        # Standard methods
        print(f"lat: {road.origin_lat}")
        print(f"lon: {road.origin_lon}")
        print(f"Saved llt2 at: {road.save_lanelet2_map(result_dir)}")
        print(f"Saved xodr at: {road.save_opendrive_map(result_dir)}")

        # Custom methods
        print(f"Original llt2 file: {road.lanelet2_map_file}")
        print(f"Original xodr file: {road.opendrive_map_file}")

        road.render(result_dir, plot_name=test_name)

    def test_mapped_road_atc(self):
        test_name = "test_mapped_road_from_api"
        result_dir = self._create_result_dir(test_name)

        road = MappedRoad(self.OPENDRIVE_MAP, self.LANELET2_MAP)

        self._stress_test(test_name, road, result_dir)

    def test_mapped_road_wrong_file(self):
        invalid_map = "doesnotexist"

        with pytest.raises(
            FileNotFoundError, match=f"File {invalid_map} does not exist"
        ):
            MappedRoad(invalid_map, self.LANELET2_MAP)

        with pytest.raises(
            FileNotFoundError, match=f"File {invalid_map} does not exist"
        ):
            MappedRoad(self.OPENDRIVE_MAP, invalid_map)

        with pytest.raises(
            FileNotFoundError, match=f"File {invalid_map} does not exist"
        ):
            MappedRoad(invalid_map, invalid_map)


if __name__ == "__main__":
    tester = TestMappedRoad()
    tester.test_mapped_road_atc()
