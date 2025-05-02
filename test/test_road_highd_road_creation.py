import matplotlib.pyplot as plt
import pytest

from pathlib import Path

from simple_scenario import LXD_AVAILABLE

if LXD_AVAILABLE:
    from lxd_io import Dataset

from simple_scenario.road import Road


class TestHighdRoadCreation:
    DATA_DIR = Path(__file__).parent / "assets/highD-dataset-v1.0"

    RESULT_DIR = Path(__file__).parent / "results" / "test_highd_road_creation"
    RESULT_DIR.mkdir(exist_ok=True)

    @pytest.mark.skipif(not LXD_AVAILABLE, reason="lxd extra is not installed")
    def test_highd_road_creation(self):
        result_dir = self.RESULT_DIR
        dataset_dir = self.DATA_DIR

        dataset = Dataset(dataset_dir)

        for recording_id in dataset.recording_ids:
            print(f"recording: {recording_id:02d} / {len(dataset.recording_ids):02d}")

            recording = dataset.get_recording(recording_id)

            speed_limit = recording.get_meta_data("speedLimit")
            if speed_limit == -1:
                speed_limit = 120

            lower_road = Road.from_highd_parameters(
                recording.get_meta_data("lowerLaneMarkings"),
                "lower",
                speed_limit=speed_limit,
            )
            upper_road = Road.from_highd_parameters(
                recording.get_meta_data("upperLaneMarkings"),
                "upper",
                speed_limit=speed_limit,
            )

            recording_background = recording.get_background_image()

            px2m_factor = 0.1
            background_image_scale_factor = dataset._background_image_scale_factor  # noqa: SLF001

            roads = {"lower": lower_road, "upper": upper_road}

            for road_name, road in roads.items():
                road.render(
                    ".",
                    str(result_dir / f"{recording_id:02d}_{road_name}_lanelets"),
                    verbose=True,
                )

                ref_line_in_image = (
                    road.ref_line / px2m_factor / background_image_scale_factor
                )
                ref_line_in_image[:, 1] = -ref_line_in_image[:, 1]

                height, width, _ = recording_background.shape
                dpi = 600
                f, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
                ax.imshow(recording_background)

                ax.plot(ref_line_in_image[:, 0], ref_line_in_image[:, 1], "r-", lw=0.1)

                for offset_line in road.offset_lines.values():
                    offset_line_in_image = (
                        offset_line / px2m_factor / background_image_scale_factor
                    )
                    offset_line_in_image[:, 1] = -offset_line_in_image[:, 1]

                    ax.plot(
                        offset_line_in_image[:, 0],
                        offset_line_in_image[:, 1],
                        "b-",
                        lw=0.1,
                    )

                ax.axis("off")
                f.savefig(
                    result_dir / f"{recording_id:02d}_{road_name}_on_background.jpg",
                    bbox_inches="tight",
                    pad_inches=0,
                )

                plt.close(f)


if __name__ == "__main__":
    tester = TestHighdRoadCreation()
    tester.test_highd_road_creation()
