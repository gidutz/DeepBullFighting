import unittest
from deep_racer_object_detection import DeepRacerObjectDetection
from object_detection.yolo import YOLO



class TestObjectDetection(unittest.TestCase):
    def test_load_yolo(self):
        # Arrange
        object_name = 'bottle'

        # Act
        detector = DeepRacerObjectDetection(object_name=object_name)

        # Assert
        self.assertIsInstance(detector.yolo, YOLO)

    def test_detect_object(self):
        # Arrange
        detector = DeepRacerObjectDetection(object_name='bottle')

        # Act

        # Assert
        self.assertIsInstance(detector.yolo, YOLO)


if __name__ == '__main__':
    unittest.main()
