import numpy as np


class DetectionResult:

    def __init__(self, box, object_class, score):
        self.box = np.array(box).astype(np.float)
        self.object_class = object_class
        self.score = score
        top, left, bottom, right = self.box
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def get_object_class(self):
        return self.object_class

    def get_box(self):
        return self.box

    def get_area(self):
        h = np.abs(self.top - self.bottom)
        w = np.abs(self.left - self.right)

        return h * w

    def get_center(self):
        x = (self.left + self.right) / 2.0
        y = (self.top + self.bottom) / 2.0

        return np.array([x, y])

    @classmethod
    def get_empty(cls):
        return cls([0, 0, 0, 0], 0, 0)

    def is_empty(self):
        return (self.box.sum() == 0) & (self.score == 0)

