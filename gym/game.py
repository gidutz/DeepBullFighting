
from deep_racer_env import DeepRacerEnv
from gym_wrappers import ObservationToDetectionWrapper, DeepRewardWrapper, ContinuesToDiscreteActionWrapper
from agents import DQNAgent
from os.path import expanduser
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    x_csrf_token = 'IjJkNjU0NzIwOWZmMDVmZGUzNTExYzQ3MjFmNWFlMzczYmVkYTRhMjEi.Xc1IOg.RUyaPAms-Ka6WTBIEJiv6xNA4-4'
    cookie = 'session=eyJjc3JmX3Rva2VuIjp7IiBiIjoiTW1RMk5UUTNNakE1Wm1Zd05XWmtaVE0xTVRGak5EY3lNV1kxWVdVek56TmlaV1JoTkdFeU1RPT0ifX0.Xc0dnA.ZZ2GmUyBYBlKC6Pv2_ny1a-yZUw; deepracer_token=0e256035-578b-4245-ab71-59c0d9e986b1'
    host = '172.20.1.54'
    model_data_dir = expanduser('~/PycharmProjects/DeepBullFighter/gym/object_detection')
    racer_env = DeepRacerEnv(x_csrf_token=x_csrf_token, cookie=cookie, host=host, model_data_dir=model_data_dir)
    racer_env = ObservationToDetectionWrapper(racer_env)
    racer_env = DeepRewardWrapper(racer_env)
    racer_env = ContinuesToDiscreteActionWrapper(racer_env, action_nvec=(3,3))

    agent = DQNAgent(racer_env, gamma=0.9, exploraion_rate=0.1)

    done = False
    obs = racer_env.reset()

    for episode in  range (100):
        print('Playing episode:', episode)
        while not done:
            action = agent.act(obs)
            print(obs, action, racer_env.action(action))
            obs_next, reward, done, _ = racer_env.step(action)
            agent.feedback(obs, action, reward, obs_next)
            obs = obs_next
        racer_env.reset()

