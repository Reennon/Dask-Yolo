from typing import Tuple, List

import numpy as np

from src.detection_helpers.detection_box import DetectionBox


class IndexLoader:
    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold
        self.detections: List[DetectionBox] = []


    def load_detection_box_per_detection_per_frame(self, detection, frame_shape) -> DetectionBox:
        frame_width, frame_height = frame_shape
        _confidence_threshold = self.confidence_threshold

        # extract the class ID and confidence (i.e., probability) of the current object detection
        scores = detection[5:]

        if max(scores) < _confidence_threshold:
            return

        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # filter out weak predictions by ensuring the detected probability is greater than the minimum
        # probability
        if confidence > _confidence_threshold:
            box = detection[0:4] * np.array([
                frame_width,
                frame_height,
                frame_width,
                frame_height
            ])
            (box_center_x, box_center_y, box_width, box_height) = box.astype("int")

            # Get detected box's coordinates
            box_x = int(box_center_x - (box_width / 2))
            box_y = int(box_center_y - (box_height / 2))

            detection_box = DetectionBox(
                box_x=box_x,
                box_y=box_y,
                box_width=box_width,
                box_height=box_height,
                detection_confidence=confidence,
                detection_class_id=class_id
            )

            return detection_box


    def extract_detections(self, output_layer, image_shape: Tuple[int, int]):
        """
        Extracts detections from net's output layer on passed image shape

        Args:
            output_layer: any
                Output layer of net
            image_shape: Tuple[int, int]
                Shape of current frame, height and width correspondingly

        Returns:
            _detections: List[DetectionBox]
                Extracted detections
        """

        _confidence_threshold = self.confidence_threshold
        _detections = self.detections

        for output in output_layer:
            for detection in output:
                detection_box = self.load_detection_box_per_detection_per_frame(
                    detection=detection,
                    frame_shape=image_shape
                )

                _detections.append(detection_box)

        return _detections



