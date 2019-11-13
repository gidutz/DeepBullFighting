import unittest
from deep_racer_object_detection import DeepRacerObjectDetection
from object_detection.yolo import YOLO
from os import path
from os.path import expanduser
import yaml

with open("../tests/conf.yaml", 'r') as stream:
    try:
        conf = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

class TestObjectDetection(unittest.TestCase):


    def test_load_yolo(self):
        # Arrange
        object_name = 'bottle'

        # Act
        detector = DeepRacerObjectDetection(object_name=object_name, model_data_dir=expanduser(conf['yolo_base_path']))

        # Assert
        self.assertIsInstance(detector.model, YOLO)

    def test_detect_object(self):
        # Arrange
        from PIL import Image
        detector = DeepRacerObjectDetection(object_name='bottle', model_data_dir=expanduser(conf['yolo_base_path']))
        sample_path = expanduser(path.join(conf['samples_dir'], 'bottles.jpg'))
        observation = Image.open(sample_path)

        # Act
        detection = detector.get_detection(observation)

        # Assert
        self.assertEqual(detection.get_object_class(), 'bottle')


if __name__ == '__main__':
    unittest.main()
