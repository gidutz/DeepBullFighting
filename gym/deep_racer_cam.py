import numpy as np
import cv2
import ssl
from urllib.request import urlopen, Request
from threading import Thread, Lock


class DeepRacerCam(Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self._data = None
        self._data_lock = Lock()

    def _data_set(self, data):
        self._data_lock.acquire(True)
        self._data = data
        self._data_lock.release()

    def _get_data(self):
        self._data_lock.acquire(True)
        data = self._data
        self._data_lock.release()
        return data

    def get_image(self):
        data = self._get_data()

        if data is None:
            return None

        img = cv2.imdecode(np.fromstring(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        return img

    def run(self):
        cam_url = 'https://172.20.1.54/route?topic=/video_mjpeg'

        headers = {
            'authority': '172.20.1.54',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36',
            'accept': 'image/webp,image/apng,image/,/*;q=0.8',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'no-cors',
            'referer': 'https://172.20.1.54/home',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9,he;q=0.8',
            'cookie': 'session=eyJjc3JmX3Rva2VuIjp7IiBiIjoiWVdFM1pEWTFPR016WkRZM056aGpaVEZoT0RnNE0yWXlaamcyWXpNMk1qUXlNVEE1WVdSaFl3PT0ifX0.XcFkHQ.1q85c-OwPCH6JuldNwKC51U4yt8; deepracer_token=83b1e193-ff33-4bf8-9b79-172e3b82b087',
        }

        cam_req = Request(cam_url, headers=headers)

        buf = bytes()
        context = ssl.SSLContext()

        with urlopen(cam_req, context=context) as cam_stream:
            while True:
                buf += cam_stream.read(1024)

                a = buf.find(b'\xff\xd8')
                b = buf.find(b'\xff\xd9')

                if a != -1 and b != -1:
                    self._data_set(buf[a:b + 2])
                    buf = buf[b + 2:]