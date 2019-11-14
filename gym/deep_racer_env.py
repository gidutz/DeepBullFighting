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
    MAX_EMPTY_STEPS = 3
    MAX_THROTTLE = 0.7
    MAX_ANGLE = 0.5

    def __init__(self, x_csrf_token, cookie, host, image_size=(224, 224), model_data_dir=None):
        self.x_csrf_token = x_csrf_token
        self.cookie = cookie
        self.host = host
        self.image_size = np.array(image_size)
        self.cam = None
        self.object_detector = DeepRacerObjectDetection(image_size=self.image_size,
                                                        object_name='bottle',
                                                        model_data_dir=model_data_dir)

        self.action_space = spaces.Box(np.array([-1.0* DeepRacerEnv.MAX_ANGLE,-1.0*DeepRacerEnv.MAX_THROTTLE]),
                                       np.array([+1.0* DeepRacerEnv.MAX_ANGLE,+1.0*DeepRacerEnv.MAX_THROTTLE]),
                                       dtype=np.float32) # angle, throttle
        self.observation_space = spaces.Box(low=0, high=255, shape=self.image_size + (3,), dtype=np.uint8)
        self.latest_detection = DetectionResult.get_empty()
        self.game_over = False
        self.empty_steps = 0

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
        logging.info("Env Reset!")
        self.empty_steps = 0
        self.latest_detection = DetectionResult.get_empty()
        return self.cam.get_image()

    def is_game_over(self):
        """
        :return:
        """
        self.game_over = False
        if (self.latest_detection.is_empty()) & (self.empty_steps > DeepRacerEnv.MAX_EMPTY_STEPS):
            print("Game Over!")
            time.sleep(10)
            self.game_over = True
        elif (self.latest_detection.is_empty()) & (self.empty_steps <= DeepRacerEnv.MAX_EMPTY_STEPS):
            self.empty_steps += 1
        elif not self.latest_detection.is_empty():
            self.empty_steps = 0

        return self.game_over

    def step(self, action, step_duration=0.3):
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

        return observation, 0, self.is_game_over(), {}
