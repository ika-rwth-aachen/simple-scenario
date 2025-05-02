from pathlib import Path

from simple_scenario import Scenario
from simple_scenario.road import Road, StraightSegment, ArcSegment, ClothoidSegment


class TestOpenxExport:
    DATA_DIR = Path(__file__).parent / "assets"
    RESULT_DIR = Path(__file__).parent / "results" / "test_openx_export"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_odr_straight_road(self):
        result_dir = self.RESULT_DIR / "test_odr_straight_road"
        result_dir.mkdir(exist_ok=True)

        straight_segment = StraightSegment(1000)

        segments = [straight_segment]

        road = Road(3, 3.75, segments)

        odr = road.create_opendrive_map()

        path_to_xodr = str(result_dir / "straight.xodr")
        odr.write_xml(path_to_xodr)

    def test_odr_curved_road(self):
        result_dir = self.RESULT_DIR / "test_odr_curved_road"
        result_dir.mkdir(exist_ok=True)

        straight_segment = StraightSegment(100)
        clothoid_segment = ClothoidSegment(3)
        arc_segment = ArcSegment(1000)

        segments = [straight_segment, clothoid_segment, arc_segment]

        road = Road(3, 3.75, segments)

        odr = road.create_opendrive_map()

        path_to_xodr = str(result_dir / "curved.xodr")
        odr.write_xml(path_to_xodr)

    def test_openx_export_a_example(self):
        """
        Download result dir, place it into $ESMINI_DIR/resources and run
        esmini --osc "$ESMINI_DIR\resources\test_openx_export_a_example\a_example.xosc" --window 60 60 1024 576
        """

        result_dir = self.RESULT_DIR / "test_openx_export_a_example"
        result_dir.mkdir(exist_ok=True)

        config_file = self.DATA_DIR / "a_example.json"

        scenario = Scenario.from_config_file(config_file)

        scenario.render(result_dir)

        scenario.save(result_dir, mode="openx")

    def test_openx_export_b_example(self):
        """
        Download result dir, place it into $ESMINI_DIR/resources and run
        esmini --osc "$ESMINI_DIR\resources\test_openx_export_b_example\b_example.xosc" --window 60 60 1024 576
        """

        result_dir = self.RESULT_DIR / "test_openx_export_b_example"
        result_dir.mkdir(exist_ok=True)

        config_file = self.DATA_DIR / "b_example.json"

        scenario = Scenario.from_config_file(config_file)

        scenario.render(result_dir)

        scenario.save(result_dir, mode="openx")

    def test_openx_export_cutout_example(self):
        """
        Download result dir, place it into $ESMINI_DIR/resources and run
        esmini --osc "$ESMINI_DIR\resources\test_openx_export_b_example\b_example.xosc" --window 60 60 1024 576
        """

        result_dir = self.RESULT_DIR / "test_openx_export_cutout_example"
        result_dir.mkdir(exist_ok=True)

        config_file = self.DATA_DIR / "cutout_example.json"

        scenario = Scenario.from_config_file(config_file)

        scenario.render(result_dir)

        scenario.save(result_dir, mode="openx")


if __name__ == "__main__":
    tester = TestOpenxExport()
    tester.test_openx_export_cutout_example()
