from object_detection.detectionresult import DetectionResult
from object_detection.yolo import YOLO
from scipy.spatial.distance import euclidean


class DeepRacerReward:

    def __init__(self, img_size=(224,224), object_name=None):
        self.img_size = img_size
        self.object_name = object_name
        self.model = YOLO(model_image_size=self.img_size)

    def detect_object(self, img):
        return self.model.detect_image(img)

    def get_distance_from_img_center(self, detection):
        center = detection.get_bounding_center()
        return euclidean(self.img_size/2, center)

    def find_max_object(self, detections, object_name=None):
        """
        In a iterable of detected objects finds the one with the largest bounding box
        :param object_name: the name of the object to find
                 if None, returns any object
                 else returns on of the coco list
        :param detections: list of
        :return:
        """
        max_area = 0
        max_detection = DetectionResult([0, 0, 0, 0], 0, 0)

        for detection in detections:
            if object_name is not None and detection['object_name'] != object_name:
                continue
            current_area = detection.get_bonding_area()
            if current_area > max_area:
                max_area = current_area
                max_detection = detection

        return max_detection, max_area

    def get_reward(self, observation):
        """
        Calculates the reward, defined as:
            %coverage of the top bounding box * distance between bounding box center and image center
        :param observation: Image of the observation
        :return: The reward
        """
        detections = self.get_detections(observation)
        max_detection, max_area = self.find_max_object(detections, self.object_name)
        coverage = max_area / (self.img_size[0]*self.img_size[1])
        distance_from_center = self.get_distance_from_img_center(detections)

        return distance_from_center*coverage

    def get_detections(self, observation):
        return self.model.detect_image(observation)

    def destroy(self):
        self.model.close_session()



