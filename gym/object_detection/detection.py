import numpy as np

class Detection:

    def __init__(self, box, object_class, score):
        self.box = box
        self.object_class = object_class
        self.score = score
        top, left, bottom, right = self.box
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def get_bonding_area(self):
        h = np.abs(self.top - self.bottom)
        w = np.abs(self.left - self.right)

        return h * w

    def get_bounding_center(self):
        x = (self.left + self.right) / 2.0
        y = (self.top + self.bottom) / 2.0

        return (x, y)

