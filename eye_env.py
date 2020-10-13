"""
This code is adapted from https://github.com/chrispiech/eyeAcuity/blob/master/evaluatePolicies.py
"""

import sys
import gym
import seeding
from io import StringIO

import numpy as np
from gym import spaces
from tqdm import tqdm

import scipy.stats as stats

'''
Size is in a range from 1 through 10
'''
SLIP_P = 0.05
FLOOR = (1. / 4.)
C = 0.8


class RNG(object):
    def __init__(self):
        self.np_random = None
        self.curr_seed = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.curr_seed = seed
        return [seed]

    def choice(self, a, size=None, replace=True, p=None):
        return self.np_random.choice(a, size, replace, p)

    def randint(self, low, high=None, size=None, dtype=int):
        return self.np_random.randint(low, high, size, dtype)


def define_action_space():
    return spaces.Box(low=-5., high=5., shape=(1,), dtype=np.float32)

def define_observation_space():
    # always observe success or failures
    return spaces.Discrete(2)


class Patient(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, model, seed=None, num_iterations=10):
        self.rng = RNG()
        self.seed(seed)

        self.num_iterations = num_iterations

        log_k1 = self.reject_sample(0.3, 0.5, -.3, 1.0)
        self.k1 = np.power(log_k1, 10)

        loc_k0 = np.log10(0.7 * self.k1)
        log_k0 = self.reject_sample(loc_k0, 0.05, -10.0, log_k1)

        self.k0 = np.power(log_k0, 10)

        self.model = model

        self.action_space = define_action_space()
        self.observation_space = define_observation_space()

        self.M = 0

    def reject_sample(self, loc, scale, minV, maxV):
        while True:
            x = stats.gumbel_r.rvs(loc, scale)
            if x > minV and x < maxV:
                return x

    def respond(self, x):
        # x is size

        if x < 0:
            return self.rng.np_random.random() < FLOOR

        if self.rng.np_random.random() < SLIP_P:
            return self.rng.np_random.random() < FLOOR

        pwr = (x - self.k0) / (self.k1 - self.k0)

        # solve overflow
        try:
            expP = 1 - pow(1 - C, pwr)
            p = max(FLOOR, expP)
        except:
            p = FLOOR

        return self.rng.np_random.random() < p

    def seed(self, seed=None):
        # we use a class object, so that if we update seed here,
        # it broadcasts into everywhere
        return self.rng.seed(seed)

    def step(self, action):
        """
        action: (num_students,)

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action.shape[0] != 1:
            raise Exception("Action needs to be applied for a patient")

        obs = self.respond(action[0]) # boolean
        obs = int(obs)

        self.M += 1

        reward = 0
        done = False
        if self.M == self.num_iterations:
            done = True
            # relative error
            # error = abs(x_star - x_hat)/x_star
            reward = -np.abs(self.model.k1 - self.k1) / self.k1

        return obs, reward, done, {}

    def reset(self):
        return None

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        # https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py


class RandomAgent(object):
    def __init__(self):
        self.k1 = np.random.randn()
        self.action_space = define_action_space()

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    num_iterations = 20
    num_patients = 2

    agent = RandomAgent()

    total_rewards = []
    for _ in tqdm(range(num_patients)):
        total_rew = 0
        env = Patient(agent)
        observation = env.reset()

        for _ in range(num_iterations):  # 1000
            action = agent.act(observation, None, None)  # np.random.randint(1, 3)
            observation, reward, done, info = env.step(action)
            total_rew += reward
            if done:
                break
        total_rewards.append(total_rew)

    print("random agent gets total reward: {}".format(np.mean(total_rewards)))
