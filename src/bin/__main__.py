import time
from typing import List

import cv2
import dask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from datetime import datetime, time

from src.detection_helpers.detection_box import DetectionBox
from src.detection_helpers.detection_drawer import DetectionDrawer
from src.index_helpers.index_loader import IndexLoader
from src.net_helpers.net_helper import NetLoader
from src.video_helpers.video_loader import VideoLoader
import logging

class Extractor:
    def __init__(self):
        pass
    def main(self):
        np.random.seed(4)
        self.video_loader = VideoLoader(
            labels_input='coco.names',
        )
        self.net_loader = NetLoader(
            config='/home/stepan/Desktop/projects/AI/labs/SPC/Dask-Yolo/darknet/cfg/yolov3-spp.cfg',
            weights='/home/stepan/Desktop/projects/AI/labs/SPC/Dask-Yolo/yolov3spp/yolov3-spp.weights',
        )

        self.labels = self.video_loader.read_labels()
        self.colors = self.video_loader.read_colors()

        # define a video capture object
        vid = cv2.VideoCapture('/home/stepan/Desktop/projects/AI/labs/SPC/Dask-Yolo/videos/2.mp4')#'/home/stepan/Desktop/projects/AI/labs/SPC/Dask-Yolo/videos/2.mp4'
        ret, frame = vid.read()
        image = frame.copy()
        (H, W) = image.shape[:2]
        self.confidence_threshold = 0.0
        print(H,W)

        i=0
        self.frames = []
        timestamp1 = datetime.now()
        while True:
            #print(f'read freame {i}')
            ret, frame = vid.read()
            if frame is None:
               break
            i+=1
            self.frames.append(frame.copy())
        print(f'video read time {datetime.now()-timestamp1}')
        vid.release()

        i=0
        extracted = []
        paralel = 0
        if paralel:
            for frame in self.frames[:20]:
                i+=1
                image = frame.copy()
                image_shape = image.shape[:2]
                output_layer = self.net_loader.read_output_layer(image=image)
                extracted_frame = self.extract_paralel(output_layer,frame)
                print(f'{i} franme extract time {datetime.now() - timestamp1}')
                extracted.append(extracted_frame)

            print(len(extracted))
            extracted2 = dask.compute(*extracted)
        else:
            for frame in self.frames[:20]:
                i += 1

                image = frame.copy()
                image_shape = image.shape[:2]
                output_layer = self.net_loader.read_output_layer(image=image)
                extracted_frame = self.extract(output_layer, frame)
                print(f'{i} franme extract time {datetime.now() - timestamp1}')
                extracted.append(extracted_frame)

            print(len(extracted))
            extracted2 = dask.compute(*extracted)
        print(f'video exract all time {datetime.now() - timestamp1}')

    @dask.delayed
    def extract_paralel(self,output_layer,frame):


        index_loader = IndexLoader(confidence_threshold=self.confidence_threshold)
        frame_detections: List[DetectionBox] = index_loader.extract_detections(
            output_layer=output_layer,
            image_shape=(720, 1280)
        )


        detection_drawer = DetectionDrawer(
            colors=self.colors,
            labels=self.labels,
        )

        fr = detection_drawer.draw_detections(
            detections=frame_detections,
            frame=frame
        )
        return fr

    def extract(self,output_layer,frame):


        index_loader = IndexLoader(confidence_threshold=self.confidence_threshold)
        frame_detections: List[DetectionBox] = index_loader.extract_detections(
            output_layer=output_layer,
            image_shape=(720, 1280)
        )


        detection_drawer = DetectionDrawer(
            colors=self.colors,
            labels=self.labels,
        )

        fr = detection_drawer.draw_detections(
            detections=frame_detections,
            frame=frame
        )
        return fr

if __name__ == '__main__':
    extractor = Extractor()
    extractor.main()

