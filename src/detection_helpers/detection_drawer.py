import cv2

from src.detection_helpers.detection_box import DetectionBox


class DetectionDrawer:
    def __init__(self, colors, labels):
        self.colors = colors
        self.labels = labels

    def draw_detections(self, detections: list[DetectionBox], frame):
        detections = list(filter(lambda detection: detection is not None, detections))

        if len(detections) <= 0:
            return

        for detection_box in detections:
            self.draw_detection_per_frame(
                detection_box=detection_box,
                frame=frame
            )

    def draw_detection_per_frame(self, detection_box: DetectionBox, frame):
        _colors = self.colors
        _labels = self.labels

        color = [int(color) for color in _colors[detection_box.detection_class_id]]

        cv2.rectangle(
            frame,
            (
                detection_box.box_x,
                detection_box.box_y,
            ),
            (
                detection_box.box_x + detection_box.box_width,
                detection_box.box_y + detection_box.box_height,
            ),
            color,
            2
        )

        label = _labels[detection_box.detection_class_id]
        confidence = round(detection_box.detection_confidence, 4)

        text = f'{label} {confidence:.2f}'

        cv2.putText(
            frame,
            text,
            (
                detection_box.box_x,
                detection_box.box_y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )