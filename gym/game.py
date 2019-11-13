
from deep_racer_env import DeepRacerEnv
from gym_wrappers import ObservationToDetectionWrapper, DeepRewardWrapper, ContinuesToDiscreteActionWrapper
from agents import DQNAgent
from os.path import expanduser

if __name__ == '__main__':
    x_csrf_token = 'IjNlZTM2OWRkY2RhZDVmMTkyMzQxZmU3MzQzNGVkMDc4MWEzMDFhNDUi.Xcwppg.n89knAPzgokH12XpYsj5TBk7LwI'
    cookie = 'session=eyJjc3JmX3Rva2VuIjp7IiBiIjoiTTJWbE16WTVaR1JqWkdGa05XWXhPVEl6TkRGbVpUY3pORE0wWldRd056Z3hZVE13TVdFME5RPT0ifX0.XcwpnQ.7vyPjIn6PF4XbjmPJc5cJ5ZMct4; deepracer_token=2c18a784-d852-44f9-9c5c-06370484051d'
    host = '172.20.1.54'
    model_data_dir = expanduser('~/PycharmProjects/DeepBullFighter/gym/object_detection')
    racer_env = DeepRacerEnv(x_csrf_token=x_csrf_token, cookie=cookie, host=host, model_data_dir=model_data_dir)
    racer_env = ObservationToDetectionWrapper(racer_env)
    racer_env = DeepRewardWrapper(racer_env)
    racer_env = ContinuesToDiscreteActionWrapper(racer_env, action_nvec=(3,3))

    agent = DQNAgent(racer_env, gamma=0.9, exploraion_rate=0.1)

    done = False
    obs = racer_env.reset()

    while not done:
        action = agent.act(obs)
        print(obs, action, racer_env.action(action))
        obs_next, reward, done, _ = racer_env.step(action)
        agent.feedback(obs, action, reward, obs_next)
