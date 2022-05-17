import numpy as np


class VideoLoader:
    def __init__(self, labels_input: str):
        self.labels_input = labels_input
        self.labels: list[str] = []

    def read_labels(self) -> list[str]:
        labels: list[str] = open(self.labels_input)\
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
