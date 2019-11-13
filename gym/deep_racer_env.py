import numpy as np
import gym
from gym import spaces
import requests
import json
import time
from deep_racer_cam import DeepRacerCam
import logging
from deep_racer_object_detection import DeepRacerObjectDetection
from object_detection.detectionresult import DetectionResult


class DeepRacerEnv(gym.Env):
    START = 'start'
    STOP = 'stop'

    def __init__(self, x_csrf_token, cookie, host, image_size=(224, 224)):
        self.x_csrf_token = x_csrf_token
        self.cookie = cookie
        self.host = host
        self.image_size = np.array(image_size)
        self.cam = None
        self.object_detector = DeepRacerObjectDetection(img_size=self.image_size,
                                                        object_name='bottle')

        self.action_space = spaces.Box(np.array([-0.9,-1.0]), np.array([+0.9,+1.0]), dtype=np.float32) # angle, throttle
        self.observation_space = spaces.Box(low=0, high=255, shape=self.image_size + (3,), dtype=np.uint8)
        self.latest_detection = DetectionResult.get_empty()

    def get_headers(self):
        headers = {
            'authority': self.host,
            'x-requested-with': 'XMLHttpRequest',
            'origin': 'https://{}'.format(self.host),
            'x-csrf-token': self.x_csrf_token,
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36',
            'content-type': 'application/json;charset=UTF-8',
            'accept': '*/*',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'cors',
            'referer': 'https://{}/home'.format(self.host),
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9,he;q=0.8',
            'cookie': self.cookie,
        }

        return headers

    def move(self, angle, throttle):
        url = "https://" + self.host + '/api/manual_drive'
        data = json.dumps({"angle": angle, "throttle": throttle})
        headers = self.get_headers()
        response = requests.put(url, headers=headers, data=data, verify=False)
        return response

    def start_stop(self, start_stop='start'):
        url = "https://" + self.host + '/api/start_stop'
        headers = self.get_headers()
        data = json.dumps({"start_stop": start_stop})
        response = requests.put(url, headers=headers, data=data, verify=False)
        return response

    def start_riding(self):
        return self.start_stop(DeepRacerEnv.START)

    def stop_riding(self):
        return self.start_stop(DeepRacerEnv.STOP)

    def reset(self):
        self.cam = DeepRacerCam(self.host, self.cookie, self.image_size)
        self.cam.start()
        logging.info("Started Game!")
        return self.cam.get_image()

    def is_game_over(self, detections):
        """

        :return:
        """

    def step(self, action, step_duration=1):
        """
        Advances the car for step_duration second
        :param action: iterable of (angle (float), throttle(float))
        :param step_duration (float or int): number of seconds to ride
        :return: next observation, reward, over, info
        """
        self.start_riding()
        self.move(*action)
        time.sleep(step_duration)
        self.stop_riding()
        observation = self.cam.get_image()
        self.latest_detection = self.object_detector.get_detection(observation)

        reward = self.object_detector.get_reward(observation)

        info = {'distance_from_center': self.object_detector.get_distance_from_img_center(),
                'box': self.object_detector.get_distance_from_img_center()}

        return observation, reward, False, info
