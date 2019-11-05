from deep_racer_env import DeepRacerEnv
from tensorflow.keras.applications import VGG16

if __name__ == '__main__':
    x_csrf_token = 'ImFhN2Q2NThjM2Q2Nzc4Y2UxYTg4ODNmMmY4NmMzNjI0MjEwOWFkYWMi.XcHBVw.CVXX1tdgal31VqlQIoqIEZe1nBE'
    cookie = 'session=eyJjc3JmX3Rva2VuIjp7IiBiIjoiWVdFM1pEWTFPR016WkRZM056aGpaVEZoT0RnNE0yWXlaamcyWXpNMk1qUXlNVEE1WVdSaFl3PT0ifX0.XcFkHQ.1q85c-OwPCH6JuldNwKC51U4yt8; deepracer_token=840ab528-4a07-445d-ae71-bcb4adac17a2'
    host = '172.20.1.54'

    vgg = VGG16()
    racer_env = DeepRacerEnv(x_csrf_token, cookie, host)

    obs = racer_env.reset()
    while True:
        object = vgg.predict(obs.reshape(1, 224, 224, 3))[0].argmax()
        if object == 737:
            speed = -.95
        else:
            speed = 0
        obs, _, _, _ = racer_env.step([0.0, speed])

    print(obs)
