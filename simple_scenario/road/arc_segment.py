import numpy as np

from .road_segment import RoadSegment


class ArcSegment(RoadSegment):
    def __init__(self, length: float = 100, radius: float = 900) -> None:
        """
        length: Length of the road segment in m
        radius: Curve radius of reference line in m.
        """
        self._config = {"length": length, "radius": radius}

        super().__init__()

        if radius < 900:
            msg = f"The radius of an arc segment must at least be 900m. Now it is: {radius}"
            raise ValueError(msg)

        self._length = length
        self._radius = radius

        self._center_x = None
        self._center_y = None
        self._start_angle = None
        self._end_angle = None
        self._sampling_angle = None

        self._curvature = None

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} length={self._length}, radius={self._radius}>"
        )

    @property
    def config(self) -> dict:
        return self._config

    @property
    def length(self) -> float:
        return self._length

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def curvature(self) -> float:
        if self._curvature is None:
            msg = (
                "Need to call compute_ref_line() first by adding the segment to a road."
            )
            raise RuntimeError(msg)

        return self._curvature

    @staticmethod
    def create_arc(
        center_x: float,
        center_y: float,
        radius: float,
        start_angle: float,
        end_angle: float,
        sampling_angle: float = np.rad2deg(1),
    ) -> np.ndarray:
        angles = np.arange(start_angle, end_angle + sampling_angle, sampling_angle)

        x = center_x + radius * np.cos(angles)
        y = center_y + radius * np.sin(angles)

        arc_points = np.array([x, y]).transpose()

        return arc_points

    def compute_ref_line(
        self, start_x: float, start_y: float, start_heading: float
    ) -> np.ndarray:
        full_circle_distance = 2 * np.pi * self._radius
        angle_dist = (self._length / full_circle_distance) * 2 * np.pi
        sampling_angle = (self._sampling_distance / full_circle_distance) * 2 * np.pi

        center_x = start_x + self._radius * np.cos(start_heading + np.pi / 2)
        center_y = start_y + self._radius * np.sin(start_heading + np.pi / 2)

        start_angle = (-np.pi / 2) + start_heading
        end_angle = start_angle + angle_dist

        ref_line = ArcSegment.create_arc(
            center_x, center_y, self._radius, start_angle, end_angle, sampling_angle
        )

        self._center_x = center_x
        self._center_y = center_y
        self._start_angle = start_angle
        self._end_angle = end_angle
        self._sampling_angle = sampling_angle

        self._curvature = 1 / self._radius

        self._ref_line = ref_line
        self._ref_line_curvature = self._curvature * np.ones_like(ref_line[:, 0])
        self._ref_line_heading = (
            np.arange(start_angle, end_angle + sampling_angle, sampling_angle)
            + np.pi / 2
        )

        return ref_line

    def compute_offset_line(self, lateral_offset: float) -> np.ndarray:
        offset_line = ArcSegment.create_arc(
            self._center_x,
            self._center_y,
            self._radius + lateral_offset,
            self._start_angle,
            self._end_angle,
            self._sampling_angle,
        )

        self._offset_lines[lateral_offset] = offset_line

        return offset_line
