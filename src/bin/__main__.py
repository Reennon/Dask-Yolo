import time
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from datetime import datetime, time

from src.detection_helpers.detection_box import DetectionBox
from src.detection_helpers.detection_drawer import DetectionDrawer
from src.index_helpers.index_loader import IndexLoader
from src.net_helpers.net_helper import NetLoader
from src.video_helpers.video_loader import VideoLoader



def main():
    np.random.seed(4)
    video_loader = VideoLoader(
        labels_input='coco.names',
    )
    net_loader = NetLoader(
        config='../darknet/cfg/yolov3-tiny.cfg',
        weights='../yolov3spp/YOLOv3-tiny.weights',
    )

    labels = video_loader.read_labels()
    colors = video_loader.read_colors()

    # define a video capture object
    vid = cv2.VideoCapture('../2.mp4')

    confidence_threshold = 0.0

    while True:

        ret, frame = vid.read()

        image = frame.copy()
        image_shape = image.shape[:2]

        output_layer = net_loader.read_output_layer(image=image)

        index_loader = IndexLoader(confidence_threshold=confidence_threshold)
        frame_detections: List[DetectionBox] = index_loader.extract_detections(
            output_layer=output_layer,
            image_shape=image_shape
        )

        detection_drawer = DetectionDrawer(
            colors=colors,
            labels=labels,
        )
        detection_drawer.draw_detections(
            detections=frame_detections,
            frame=image
        )

        # Display the resulting frame
        cv2.imshow('AI', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
