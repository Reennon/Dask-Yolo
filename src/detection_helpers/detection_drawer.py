import cv2
from typing import List

import dask

from src.detection_helpers.detection_box import DetectionBox


class DetectionDrawer:
    def __init__(self, colors, labels):
        self.colors = colors
        self.labels = labels

    @dask.delayed
    def draw_detection(self, id: str, frame, class_id, confidence, box):
        (x, y) = (box[0], box[1])
        (w, h) = (box[2], box[3])

        color = [int(c) for c in self.colors[class_id]]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "{}|{}: {:.4f}".format(id, self.labels[class_id], confidence)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    def draw_detections(self, detections: List[DetectionBox], frame):
        idxs, boxes, confidences, class_ids = detections

        if len(idxs) > 0:
            # loop over the indexes we are keeping
            drawn_detections = []
            for i in idxs.flatten():
                # extract the bounding box coordinates
                class_id = class_ids[i]
                confidence = confidences[i]
                box = boxes[i]
                drawn_detection = self.draw_detection(
                    id=i,
                    class_id=class_id,
                    confidence=confidence,
                    box=box,
                    frame=frame
                )

                drawn_detections.append(drawn_detection)

            dask.compute(*drawn_detections)

        return frame


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