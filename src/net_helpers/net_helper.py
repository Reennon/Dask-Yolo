from copy import copy

import cv2
import dask


class NetLoader:
    def __init__(self, config: str, weights: str):
        self.config: str = config
        self.weights: str = weights
        self.net = cv2.dnn.readNetFromDarknet(
            self.config,
            self.weights
        )

    def read_output_layer(self, image):
        _net = self.net

        layer_names = _net.getLayerNames()

        layer_names = [layer_names[i - 1] for i in _net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            (416, 416),
            swapRB=True,
            crop=False
        )
        _net.setInput(blob)

        output_layer = _net.forward(layer_names)

        return output_layer
