from __future__ import annotations

import numpy as np

from pyclothoids import Clothoid

from .road_segment import RoadSegment


class ClothoidSegment(RoadSegment):
    def __init__(self, r: float = 3) -> None:
        """
        r: Kennstelle of clothoid. Should always be between 1 and 3. 1: easiest to drive, 3: hardest to drive
        """
        self._config = {"r": r}

        super().__init__()

        if not (1 <= r <= 3):
            msg = f"The clothoid parameter 'r' must be between 1 and 3. Now it is: {r}"
            raise ValueError(msg)

        self._r = r

        self._ref_clothoid = None
        self._offset_clothoids = {}

        # State
        self._clothoid_start_curvature = None
        self._clothoid_end_curvature = None
        self._ref_line = None
        self._ref_line_length = None

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} r={self._r}>"

    @property
    def config(self) -> dict:
        return self._config

    @property
    def r(self) -> float:
        return self._r

    @property
    def ref_clothoid(self) -> Clothoid:
        return self._ref_clothoid

    @property
    def offset_clothoids(self) -> list[Clothoid]:
        return self._offset_clothoids

    @property
    def curvature_start(self) -> float:
        if self._clothoid_start_curvature is None:
            msg = (
                "Need to call compute_ref_line() first by adding the segment to a road."
            )
            raise RuntimeError(msg)

        return self._clothoid_start_curvature

    @property
    def curvature_end(self) -> float:
        if self._clothoid_end_curvature is None:
            msg = (
                "Need to call compute_ref_line() first by adding the segment to a road."
            )
            raise RuntimeError(msg)

        return self._clothoid_end_curvature

    @property
    def length(self) -> float:
        if self._ref_line_length is None:
            msg = (
                "Need to call compute_ref_line() first by adding the segment to a road."
            )
            raise RuntimeError(msg)

        return self._ref_line_length

    def compute_ref_line(
        self, start_x: float, start_y: float, start_heading: float, end_radius: float
    ) -> np.ndarray:
        clothoid_param_A = end_radius / self.r  # noqa: N806
        clothoid_length_L = (clothoid_param_A**2) / end_radius  # noqa: N806

        self._clothoid_start_curvature = 0.0
        self._clothoid_end_curvature = 1 / end_radius

        curvature_rate = (
            self._clothoid_end_curvature - self._clothoid_start_curvature
        ) / clothoid_length_L

        clothoid = Clothoid.StandardParams(
            start_x,
            start_y,
            start_heading,
            self._clothoid_start_curvature,
            curvature_rate,
            clothoid_length_L,
        )

        clothoid_x = []
        clothoid_y = []

        for s in np.arange(
            0, clothoid_length_L + self._sampling_distance, self._sampling_distance
        ):
            clothoid_x.append(clothoid.X(s))
            clothoid_y.append(clothoid.Y(s))

        ref_line = np.array([clothoid_x, clothoid_y]).transpose()

        self._ref_clothoid = clothoid
        self._ref_line = ref_line

        clothoid_curvature = [0.0]
        for _ in np.arange(
            self._sampling_distance,
            clothoid.length + self._sampling_distance,
            self._sampling_distance,
        ):
            clothoid_curvature.append(
                clothoid_curvature[-1] + curvature_rate * self._sampling_distance
            )
        self._ref_line_curvature = clothoid_curvature

        heading = [
            clothoid.Theta(s)
            for s in np.arange(
                0, clothoid.length + self._sampling_distance, self._sampling_distance
            )
        ]

        self._ref_line_heading = heading

        self._ref_line_length = float(
            np.sum(
                np.sqrt(
                    np.diff(self._ref_line[:, 0]) ** 2
                    + np.diff(self._ref_line[:, 1]) ** 2
                )
            )
        )

        return ref_line

    def compute_offset_line(self, lateral_offset: float) -> np.ndarray:
        clothoid_start_x = self._ref_line[0, 0] + lateral_offset * np.cos(
            self._ref_clothoid.ThetaStart - np.pi / 2
        )
        clothoid_start_y = self._ref_line[0, 1] + lateral_offset * np.sin(
            self._ref_clothoid.ThetaStart - np.pi / 2
        )

        clothoid_end_x = self._ref_line[-1, 0] + lateral_offset * np.cos(
            self._ref_clothoid.ThetaEnd - np.pi / 2
        )
        clothoid_end_y = self._ref_line[-1, 1] + lateral_offset * np.sin(
            self._ref_clothoid.ThetaEnd - np.pi / 2
        )

        clothoid = Clothoid.G1Hermite(
            clothoid_start_x,
            clothoid_start_y,
            self._ref_clothoid.ThetaStart,
            clothoid_end_x,
            clothoid_end_y,
            self._ref_clothoid.ThetaEnd,
        )

        # Make sure there are exactly the same amount of points than in ref_line
        n_ref_clothoids_points = self._ref_line.shape[0]
        offset_line = np.array(clothoid.SampleXY(n_ref_clothoids_points)).transpose()

        self._offset_clothoids[lateral_offset] = clothoid
        self._offset_lines[lateral_offset] = offset_line

        return offset_line
