import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime
from deep_racer_env import DeepRacerEnv
from gym_wrappers import ObservationToDetectionWrapper, DeepRewardWrapper, ContinuesToDiscreteActionWrapper
from agents import DQNAgent
from os.path import expanduser
DATE_FORMAT = '%Y%m%d-%M%S'

if __name__ == '__main__':
    x_csrf_token = 'IjRjOTcwOGM5ZTFjOGJjMmQ4NTUwYTVlYWYyYWU3NWE0Y2MwZDRmMTQi.XdK-Dg.-6lcHQDEbaRs3HtEbhSbwAAwuEQ'
    cookie = 'session=eyJjc3JmX3Rva2VuIjp7IiBiIjoiTkdNNU56QTRZemxsTVdNNFltTXlaRGcxTlRCaE5XVmhaakpoWlRjMVlUUmpZekJrTkdZeE5BPT0ifX0.XdK9_g.Uy9HA2sMiYKp1HVK182xL1OvIMw; deepracer_token=1b0fcc24-d89d-47d1-a25e-b3ca391acb1b'
    host = '172.20.1.54'
    model_data_dir = expanduser('~/PycharmProjects/DeepBullFighter/gym/object_detection')
    racer_env = DeepRacerEnv(x_csrf_token=x_csrf_token, cookie=cookie, host=host, model_data_dir=model_data_dir)
    racer_env = ObservationToDetectionWrapper(racer_env)
    racer_env = DeepRewardWrapper(racer_env)
    racer_env = ContinuesToDiscreteActionWrapper(racer_env, action_nvec=(3,3))

    agent = DQNAgent(racer_env, gamma=0.9, exploraion_rate=0.1)

    done = False
    obs = racer_env.reset()
    game_ts = datetime.utcnow().strftime(DATE_FORMAT)
    for episode in  range (100):
        print('Playing episode:', episode)
        rewards = []
        while not done:
            action = agent.act(obs)
            print(obs, action, racer_env.action(action))
            obs_next, reward, done, _ = racer_env.step(action)
            agent.feedback(obs, action, reward, obs_next)
            obs = obs_next
            rewards.append(reward)
        racer_env.reset()
        done = False
        print("Total Reward: ", np.sum(rewards))
        with open('./agents_data/{}.txt'.format(game_ts), 'w+') as file:
            file.write(str(rewards))
            file.write("\n")
