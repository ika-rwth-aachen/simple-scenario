import numpy as np

from abc import abstractmethod


class RoadSegment:
    def __init__(self, sampling_distance: float = 0.5) -> None:
        """
        sampling_distance: Sampling distance along s in m
        """

        self._sampling_distance = sampling_distance

        self._ref_line = None
        self._ref_line_curvature = None
        self._ref_line_heading = None
        self._offset_lines = {}

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def sampling_distance(self) -> float:
        return self._sampling_distance

    @property
    def ref_line(self) -> np.ndarray:
        return self._ref_line

    @property
    def ref_line_curvature(self) -> np.ndarray:
        return self._ref_line_curvature

    @property
    def ref_line_heading(self) -> np.ndarray:
        return self._ref_line_heading

    @property
    def offset_lines(self) -> dict:
        return self._offset_lines

    @abstractmethod
    def compute_ref_line(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def compute_offset_line(self, lateral_offset: float) -> np.ndarray:
        raise NotImplementedError
