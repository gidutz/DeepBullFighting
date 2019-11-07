import requests
import json
import time
from deep_racer_cam import DeepRacerCam
import logging
from deep_racer_reward import DeepRacerReward

class DeepRacerEnv:
    START = 'start'
    STOP = 'stop'

    def __init__(self, x_csrf_token, cookie, host, img_size=(224,224)):
        self.x_csrf_token = x_csrf_token
        self.cookie = cookie
        self.host = host
        self.cam = None
        self.reward_calculator = DeepRacerReward(img_size=img_size,
                                                 object_name='bottle')

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
        self.cam = DeepRacerCam(self.host, self.cookie)
        self.cam.start()
        time.sleep(1)
        logging.info("Started Game!")
        return self.cam.get_image()


    def step(self, action):
        """
        Advances the car for 1 second
        :param action: iterable of (angle (float), throttle(float))
        :return: next observation, reward, over, info
        """
        self.start_riding()
        self.move(*action)
        time.sleep(1)
        self.stop_riding()
        observation = self.cam.get_image()
        reward = self.reward_calculator.get_reward(observation)

        info = {'distance_from_center': self.reward_calculator.get_distance_from_img_center(),
                'box': self.reward_calculator.get_distance_from_img_center()}

        return observation, reward, False, info
