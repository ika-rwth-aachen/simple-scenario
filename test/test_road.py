import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from simple_scenario.road import Road, StraightSegment, ArcSegment, ClothoidSegment


class TestRoad:
    RESULT_DIR = Path(__file__).parent / "results" / "test_road"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_straight_road(self):
        result_dir = self.RESULT_DIR / "test_straight_road"
        result_dir.mkdir(exist_ok=True)

        straight_segment = StraightSegment(100)

        road = Road(3, 3.75, [straight_segment], goal_position=80)

        plot_name = "short_road"

        road.render(result_dir, plot_name=f"{plot_name}_default")
        road.render(
            result_dir,
            plot_name=f"{plot_name}_no-lanelet_verbose",
            use_lanelet=False,
            verbose=True,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_no-lanelet_no-verbose",
            use_lanelet=False,
            verbose=False,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_lanelet_no-verbose",
            use_lanelet=True,
            verbose=False,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_lanelet_verbose",
            use_lanelet=True,
            verbose=True,
        )

        road.render_curvature(result_dir, plot_name=plot_name)
        road.render_heading(result_dir, plot_name=plot_name)

    def test_straight_road_at_angle(self):
        result_dir = self.RESULT_DIR / "test_straight_road_at_angle"
        result_dir.mkdir(exist_ok=True)

        straight_segment = StraightSegment(100, heading=np.pi / 4)

        road = Road(3, 3.75, [straight_segment])

        plot_name = "short_road_at_angle"

        road.render(result_dir, plot_name=f"{plot_name}_default")
        road.render(result_dir, plot_name=f"{plot_name}_default_clean", clean=True)
        road.render(
            result_dir,
            plot_name=f"{plot_name}_no-lanelet_verbose",
            use_lanelet=False,
            verbose=True,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_no-lanelet_no-verbose",
            use_lanelet=False,
            verbose=False,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_lanelet_no-verbose",
            use_lanelet=True,
            verbose=False,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_lanelet_verbose",
            use_lanelet=True,
            verbose=True,
        )

        road.render_curvature(result_dir, plot_name=plot_name)
        road.render_heading(result_dir, plot_name=plot_name)

    def test_curved_road(self):
        result_dir = self.RESULT_DIR / "test_curved_road"
        result_dir.mkdir(exist_ok=True)

        straight_segment = StraightSegment(100)
        clothoid_segment = ClothoidSegment(3)
        arc_segment = ArcSegment(1000)

        segments = [straight_segment, clothoid_segment, arc_segment]

        road = Road(2, 3.75, segments)

        plot_name = "curved_road"

        road.render(result_dir, plot_name=f"{plot_name}_default")
        road.render(result_dir, plot_name=f"{plot_name}_default_clean", clean=True)
        road.render(
            result_dir,
            plot_name=f"{plot_name}_no-lanelet_verbose",
            use_lanelet=False,
            verbose=True,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_no-lanelet_no-verbose",
            use_lanelet=False,
            verbose=False,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_lanelet_no-verbose",
            use_lanelet=True,
            verbose=False,
        )
        road.render(
            result_dir,
            plot_name=f"{plot_name}_lanelet_verbose",
            use_lanelet=True,
            verbose=True,
        )

        road.render_curvature(result_dir, plot_name=plot_name)
        road.render_heading(result_dir, plot_name=plot_name)

    def test_precise_plot(self):
        result_dir = self.RESULT_DIR / "test_precise_plot"
        result_dir.mkdir(exist_ok=True)

        straight_segment = StraightSegment(1000)

        road = Road(3, 3.75, [straight_segment], goal_position=80)

        plot_name = "long_road"

        # Render directly to file
        road.render(result_dir, plot_name=f"{plot_name}")
        road.render(result_dir, plot_name=f"{plot_name}_clean", clean=True)

        f, ax = plt.subplots()
        road.render(plot_dir_or_ax=ax)
        f.savefig(result_dir / "long_road_ax")

        road.render_curvature(result_dir, plot_name)
        road.render_heading(result_dir, plot_name)


if __name__ == "__main__":
    test_road = TestRoad()
    test_road.test_curved_road()
