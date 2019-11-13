from object_detection.detectionresult import DetectionResult
from object_detection.yolo import YOLO
from scipy.spatial.distance import euclidean


class DeepRacerObjectDetection:

    def __init__(self, image_size=(224, 224), object_name=None, model_data_dir=None):
        """

        :param image_size: tuple, (width, height) of the image input
        :param object_name: name of the object to search
        """
        self.image_size = image_size
        self.object_name = object_name
        self.model = YOLO(model_image_size=tuple(self.image_size), base_dir=model_data_dir)

    def detect_object(self, img):
        return self.model.detect_image(img)

    def get_distance_from_img_center(self, detection):
        center = detection.get_center()

        return euclidean(self.image_size / 2, center)

    def _find_max_object(self, detections):
        """
        In a iterable of detected objects finds the one with the largest bounding box
        :param object_name: the name of the object to find
                 if None, returns any object
                 else returns on of the coco list
        :param detections: list of
        :return: max_detection: Detection with largest area
                max_area: Area of the max detection
        """
        max_area = 0
        max_detection = DetectionResult.get_empty()

        for detection in detections:
            if self.object_name is not None and detection.get_object_class() != self.object_name:
                continue
            current_area = detection.get_area()
            if current_area > max_area:
                max_area = current_area
                max_detection = detection

        return max_detection

    def get_detection(self, observation):
        return self._find_max_object(self.model.detect_image(observation))

    def destroy(self):
        self.model.close_session()



