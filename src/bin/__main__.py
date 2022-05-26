import time
from typing import List

import cv2
import dask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from datetime import datetime, time

from dask import delayed, visualize

from src.detection_helpers.detection_box import DetectionBox
from src.detection_helpers.detection_drawer import DetectionDrawer
from src.index_helpers.index_loader import IndexLoader
from src.net_helpers.net_helper import NetLoader
from src.video_helpers.video_loader import VideoLoader
from dask.distributed import Client
import logging


class Extractor:
    def __init__(self):
        self.confidence_threshold = None
        self.frames = None
        self.colors = None
        self.labels = None
        self.net_loader = None
        self.video_loader = None


    def main(self):
        #client = Client(n_workers=12)
        np.random.seed(4)
        self.video_loader = VideoLoader(
            labels_input='./src/bin/coco.names',
        )
        chunk = 1
        self.net_loader = [NetLoader(
            config='./darknet/cfg/yolov3-spp.cfg',
            weights='./weights/yolov3spp/yolov3-spp.weights',
        ) for _ in range(chunk)]

        self.labels = self.video_loader.read_labels()
        self.colors = self.video_loader.read_colors()

        # define a video capture object
        vid = cv2.VideoCapture('./videos/2.mp4')
        ret, frame = vid.read()
        image = frame.copy()
        (H, W) = image.shape[:2]
        self.confidence_threshold = 0.0
        print(H, W)

        i = 0
        self.frames = []
        timestamp1 = datetime.now()
        while True:
            # print(f'read freame {i}')
            ret, frame = vid.read()
            if frame is None:
                break
            i += 1
            self.frames.append(frame.copy())
        print(f'video read time {datetime.now() - timestamp1}')
        vid.release()


        parallel = 1
        if parallel:
            computed = []
            for i in range(int(len(self.frames[:300])/chunk)):
                frames = self.frames[i * chunk: (i+1) * chunk]
                print(f'{datetime.now()} Chunk: {i} Frames: {i * chunk, (i+1) * chunk}')
                j = 0
                extracted = []
                #print((i) * chunk, (i+1) * chunk)
                for frame, net in zip(frames, self.net_loader):
                    j += 1
                    image = frame.copy()
                    image_shape = image.shape[:2]

                    extracted_frame = self.extract_parallel(
                        net=net,
                        image=image,
                        frame=frame
                    )
                    print(f'{j} frame extract time {datetime.now() - timestamp1}')
                    extracted.append(extracted_frame)

                computed.extend(dask.compute(*extracted))

            print(len(extracted))
            #print(extracted.visualize())
            #visualize(*extracted)
            #extracted = dask.compute(*extracted)
        else:
            for frame in self.frames[:20]:
                i += 1

                image = frame.copy()
                image_shape = image.shape[:2]
                output_layer = self.net_loader.read_output_layer(image=image)
                extracted_frame = self.extract(output_layer, frame)
                print(f'{i} frame extract time {datetime.now() - timestamp1}')
                extracted.append(extracted_frame)

            print(len(extracted))

        VideoLoader.write_video(computed, './videos/output/output.mp4')

        print(f'video exract all time {datetime.now() - timestamp1}')

    @dask.delayed
    def extract_parallel(self, net, frame, image):
        output_layer = net.read_output_layer(image=image)
        #output_layer = dask.compute(output_layer)
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

    def extract(self, output_layer, frame):
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
    cv2.destroyAllWindows()
