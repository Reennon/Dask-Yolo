import numpy as np
from typing import List


class VideoLoader:
    def __init__(self, labels_input: str):
        self.labels_input = labels_input
        self.labels: List[str] = []

    def read_labels(self) -> List[str]:
        labels: List[str] = open(self.labels_input)\
            .read()\
            .strip()\
            .split("\n")

        self.labels = labels

        return labels


    def read_colors(self):
        _labels = self.labels

        colors = np.random.randint(
            0,
            255,
            size=(
                len(_labels),
                3
            ),
            dtype="uint8"
        )

        return colors
