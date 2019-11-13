from itertools import product
from gym import spaces
from gym import ActionWrapper, ObservationWrapper, RewardWrapper
import numpy as np

class ContinuesToDiscreteActionWrapper(ActionWrapper):
    """
    Converts continues action into discrete
    """

    def __init__(self, env, action_nvec):
        super().__init__(env)
        self.actions = list(product(*[np.linspace(env.action_space.low[i], env.action_space.high[i], n)
                                      for i, n in enumerate(action_nvec)]))
        self.action_space = spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]

    def reverse_action(self, action):
        return self.actions.index(action)


class ObservationToDetectionWrapper(ObservationWrapper):

    def observation(self, observation):

        pass


class DeepRewardV1Wrapper(RewardWrapper):

    def reward(self, detections):
        """
         Calculates the reward, defined as:
             %coverage of the top bounding box * distance between bounding box center and image center
         :param detections: Image of the observation
         :return: The reward
         """
        max_detection, max_area = self.find_max_object(detections, self.object_name)
        coverage = max_area / (self.img_size[0] * self.img_size[1])
        distance_from_center = self.get_distance_from_img_center(detections)

        return distance_from_center * coverage
