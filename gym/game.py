import numpy as np
import logging
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions

from deep_racer_env import DeepRacerEnv

if __name__ == '__main__':
    x_csrf_token = 'IjYxMGNjOGYwMjc4YmY5ZjFkMjYwNWJiNzM1NmQ0ZjNiZmUxZmU4NmQi.XcLyIg.qGVU4kJNf98jBMaz16Y2Eg068Us'
    cookie = 'session=eyJjc3JmX3Rva2VuIjp7IiBiIjoiTmpFd1kyTTRaakF5TnpoaVpqbG1NV1F5TmpBMVltSTNNelUyWkRSbU0ySm1aVEZtWlRnMlpBPT0ifX0.XcLFeg.BWhaGOuBkzyMpY1dvQoqI7jG4Ts; deepracer_token=5ba00dd2-48fd-4b85-817c-4b5e1a9e865e'
    host = '172.20.1.54'

    vgg = VGG16()
    racer_env = DeepRacerEnv(x_csrf_token, cookie, host)

    done = False
    obs = racer_env.reset()

    while not done:
        objects = [pred[1] for pred in decode_predictions(vgg.predict(np.stack([obs])), top=5)[0]]
        logging.info(objects)

        if 'pop_bottle' in objects:
            speed = - 0.8
        else:
            speed = 0

        obs, _, done, _ = racer_env.step([0.0, speed])
