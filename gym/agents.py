
import numpy as np
from random import sample

import keras.layers as lyr
from keras.models import Model
import os
from deep_racer_env import DeepRacerEnv
from gym_wrappers import ContinuesToDiscreteActionWrapper


class DQNAgent:
    WEIGHTS_PATH = './agents_data/model_weights.h5'
    def __init__(self, env: DeepRacerEnv, gamma, exploraion_rate):
        self.env = env
        self.gamma = gamma
        self.exploraion_rate = exploraion_rate
        self.memory = []
        self.model = self._build_model(hiddens=[256]*2)

    def _build_model(self, hiddens=None):

        lyr_inp = lyr.Input(self.env.observation_space.shape)
        lyr_prev = lyr_inp

        if hiddens is not None:
            for hidden_size in hiddens:
                lyr_prev = lyr.Dense(hidden_size, activation='tanh')(lyr_prev)

        lyr_out = lyr.Dense(self.env.action_space.n, activation='linear')(lyr_prev)

        model = Model(lyr_inp, lyr_out)
        model.compile(loss='mse', optimizer='nadam')

        if os.path.exists(DQNAgent.WEIGHTS_PATH):
            try:
                model.load_weights()
            except:
                print('Cannot load weights, you might want to delete the file')

        return model

    def act(self, observation):
        if np.random.rand() < self.exploraion_rate:
            return self.env.action_space.sample()

        return self.model.predict(np.stack([observation]))[0].argmax()

    def td_step(self, batch_size=32):
        batch = sample(self.memory, min(batch_size, len(self.memory)))
        X = np.stack([observation for observation, _, _, _ in batch])
        a = np.stack([action for _, action, _, _ in batch])
        r = np.stack([reward for _, _, reward, _ in batch])
        X_next = np.stack([next_observation for _, _, _, next_observation in batch])

        R = self.model.predict(X)
        R_next = self.model.predict(X_next)
        R[range(len(R)),a] = r + self.gamma * R_next.max(axis=1)
        self.model.train_on_batch(X, R)
        self.model.save_weights(DQNAgent.WEIGHTS_PATH, overwrite=True)

    def feedback(self, observation, action, reward, next_observation):
        self.memory.append((observation, action, reward, next_observation))
        self.td_step()

