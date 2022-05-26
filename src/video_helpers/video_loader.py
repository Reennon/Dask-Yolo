import datetime

import cv2
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

    @staticmethod
    def write_video(frames: list, file: str):
        writer = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
        for i in range(0, len(frames)):
            resize = cv2.resize(frames[i], (1280, 720))
            writer.write(resize)
            print(f'{datetime.datetime.now()} Written {i}th frame')
        writer.release()
