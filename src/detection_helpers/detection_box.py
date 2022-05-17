class DetectionBox:
    def __init__(self, box_x, box_y, box_width, box_height, detection_confidence, detection_class_id):
        self.box_x: int = box_x
        self.box_y: int = box_y
        self.box_width: int = box_width
        self.box_height: int = box_height
        self.detection_confidence: float = detection_confidence
        self.detection_class_id: int = detection_class_id
