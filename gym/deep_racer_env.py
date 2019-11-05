import requests
import json
import time
from deep_racer_cam import DeepRacerCam
import logging

class DeepRacerEnv():
    START = 'start'
    STOP = 'stop'

    def __init__(self, x_csrf_token, cookie, host):
        self.x_csrf_token = x_csrf_token
        self.cookie = cookie
        self.host = host
        self.cam = None

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
        self.start_riding()
        self.move(*action).text
        time.sleep(1)
        self.stop_riding()
        observation = self.cam.get_image()

        return observation, 0, False, {}
