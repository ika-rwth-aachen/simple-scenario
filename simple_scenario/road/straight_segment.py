import numpy as np

from loguru import logger

from .road_segment import RoadSegment


class StraightSegment(RoadSegment):
    def __init__(self, length: float, heading: float = 0.0) -> None:
        """
        length: Length of the road segment in m
        heading: Heading of the road segment in rad
        """

        self._config = {"length": length, "heading": heading}

        super().__init__()

        self._length = length
        self._heading = heading

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} length={self._length}>"

    @property
    def config(self) -> dict:
        return self._config

    @property
    def length(self) -> float:
        return self._length

    @property
    def heading(self) -> float:
        return self._heading

    def compute_ref_line(self, start_x: float, start_y: float) -> np.ndarray:
        logger.debug(
            "Create ref_line of straight segment starting at: {}, {}", start_x, start_y
        )

        straight_x = [start_x]
        straight_y = [start_y]

        for _ in np.arange(
            self._sampling_distance,
            self._length + self._sampling_distance,
            self._sampling_distance,
        ):
            next_straight_x = straight_x[-1] + self._sampling_distance * np.cos(
                self._heading
            )
            next_straight_y = straight_y[-1] + self._sampling_distance * np.sin(
                self._heading
            )

            straight_x.append(next_straight_x)
            straight_y.append(next_straight_y)

        ref_line = np.array([straight_x, straight_y]).transpose()

        self._ref_line = ref_line
        self._ref_line_curvature = np.zeros_like(ref_line[:, 0])
        self._ref_line_heading = self._heading * np.ones_like(ref_line[:, 0])

        return ref_line

    def compute_offset_line(self, lateral_offset: float) -> np.ndarray:
        x = []
        y = []

        for p in self._ref_line:
            x.append(p[0] + lateral_offset * np.cos(self._heading - np.pi / 2))
            y.append(p[1] + lateral_offset * np.sin(self._heading - np.pi / 2))

        offset_line = np.array([x, y]).transpose()

        self._offset_lines[lateral_offset] = offset_line

        return offset_line
