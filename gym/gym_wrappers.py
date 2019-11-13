from itertools import product
from gym import spaces
from gym import ActionWrapper, ObservationWrapper, RewardWrapper
import numpy as np

from deep_racer_env import DeepRacerEnv


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
    def __init__(self, env: DeepRacerEnv):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def observation(self, observation):
        top, left, bottom, right = self.env.latest_decetion.get_box()
        return np.array([
            top / self.env.image_size[1],
            left / self.env.image_size[0],
            bottom / self.env.image_size[1],
            right / self.env.image_size[0]
        ])



class DeepRewardWrapper(RewardWrapper):

    def reward(self, reward):
        """
         Calculates the reward, defined as:
             %coverage of the top bounding box * distance between bounding box center and image center
         :param detectios: max detection of the observation
         :return: The reward
         """
        coverage = self.env.latest_decetion.get_bounding_area() / (self.env.image_size[0] * self.env.image_size[1])
        distance_from_center = ((self.env.latest_decetion.get_bounding_center() - self.env.image_size)**2).sum()**0.5
        distance_from_center /= ((self.env.image_size / 2)**2).sum()**0.5

        return (1.0 - distance_from_center) * coverage
