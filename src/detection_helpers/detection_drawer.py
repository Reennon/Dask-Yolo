import cv2
from typing import List

from src.detection_helpers.detection_box import DetectionBox


class DetectionDrawer:
    def __init__(self, colors, labels):
        self.colors = colors
        self.labels = labels

    def draw_detections(self, detections: List[DetectionBox], frame):
        idxs , boxes, confidences,classIDs = detections
        if len(idxs) > 0:
            # loop over the indexes we are keeping

            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in self.colors[classIDs[i]]]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}|{}: {:.4f}".format(i,self.labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
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