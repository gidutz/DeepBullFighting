import numpy as np
import cv2
import ssl
from urllib.request import urlopen, Request
from threading import Thread, Lock


class DeepRacerCam(Thread):
    def __init__(self, hostname, cookie):
        super().__init__()
        self.daemon = True

        self.hostname = hostname
        self.cookie = cookie

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
        cam_url = 'https://{}/route?topic=/video_mjpeg'.format(self.hostname)

        headers = {
            'authority': self.hostname,
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36',
            'accept': 'image/webp,image/apng,image/,/*;q=0.8',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'no-cors',
            'referer': 'https://{}/home'.format(self.hostname),
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9,he;q=0.8',
            'cookie': self.cookie,
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