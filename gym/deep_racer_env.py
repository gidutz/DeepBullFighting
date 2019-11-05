import requests
import json


class DeepRacerEnv():
    START = 'start'
    STOP = 'stop'

    def __init__(self, x_csrf_token, cookie):
        self.x_csrf_token = x_csrf_token
        self.cookie = cookie
        self.hostname = 'https://172.20.1.54/'

    def get_headers(self):
        headers = {
            'authority': '172.20.1.54',
            'x-requested-with': 'XMLHttpRequest',
            'origin': 'https://172.20.1.54',
            'x-csrf-token': self.x_csrf_token,
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36',
            'content-type': 'application/json;charset=UTF-8',
            'accept': '*/*',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'cors',
            'referer': 'https://172.20.1.54/home',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9,he;q=0.8',
            'cookie': self.cookie,
        }

        return headers

    def move(self, angle, throttle):
        url = self.hostname + 'api/manual_drive'
        data = json.dumps({"angle": angle, "throttle": throttle})
        headers = self.get_headers()
        response = requests.put(url, headers=headers, data=data, verify=False)
        return response

    def start_stop(self, start_stop='start'):
        url = self.hostname + 'api/start_stop'
        headers = self.get_headers()
        data = json.dumps({"start_stop": start_stop})
        response = requests.put(url, headers=headers, data=data, verify=False)
        return response

    def start_riding(self):
        return self.start_stop(DeepRacerAgent.START)

    def stop_riding(self):
        return self.start_stop(DeepRacerAgent.STOP)

    def reset(self):
        self.stop_riding()
        self.start_riding()
