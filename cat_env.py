import sys
import gym
import seeding
from io import StringIO

import numpy as np
from gym import spaces


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


class ItemBank(object):
    def __init__(self, num_items, num_students, rng=None):
        if rng is None:
            self.rng = RNG()
            self.rng.seed(None)

        self.num_items = num_items
        self.item_difficulties = self.rng.np_random.randn(num_items)

        self.action_space = define_action_space(self.num_items, num_students)


def define_action_space(num_items, num_students):
    return spaces.MultiDiscrete([num_items] * num_students)


def define_observation_space(num_students):
    return spaces.Discrete(num_students)


class CatEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, model, item_bank, num_students=50, num_iterations=10):
        self.rng = RNG()
        self.seed()

        self.num_items = item_bank.num_items
        self.num_students = num_students
        self.num_iterations = num_iterations

        # reset does not change these
        self.item_difficulties = item_bank.item_difficulties
        self.student_abilities = self.rng.np_random.randn(num_students)

        self.model = model
        self.M = 0

        self.action_space = item_bank.action_space
        self.observation_space = define_observation_space(num_students)

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
        if action.shape[0] != self.num_students:
            raise Exception("Action needs to be specified for each student")

        if np.max(action) >= self.num_items:
            raise Exception("Can't choose an action based on index {}".format(np.max(action)))

        done = False

        correct_prob = sigmoid(self.item_difficulties[action] - self.student_abilities)
        corrects = self.rng.np_random.binomial(1, correct_prob)  # correct_prob.shape[0]

        obs = corrects

        self.M += 1

        reward = 0
        if self.M == self.num_iterations:
            done = True
            # MSE (mean square error)
            reward = np.mean(np.square(self.model.student_abilities - self.student_abilities))

        return obs, reward, done, {}

    def reset(self):
        # TODO: what is the initial observation?
        return None

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        # https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py


class RandomAgent(object):
    def __init__(self, num_students, item_bank):
        self.student_abilities = np.random.randn(num_students)
        self.item_bank = item_bank
        self.num_items = item_bank.num_items

    def act(self, observation, reward, done):
        return self.item_bank.action_space.sample()


if __name__ == '__main__':
    pass
    num_iterations = 10

    item_bank = ItemBank(num_items=100, num_students=50)
    agent = RandomAgent(num_students=50, item_bank=item_bank)
    env = CatEnv(agent, item_bank, num_students=50, num_iterations=num_iterations)

    num_exp = 10
    total_rewards = []
    for _ in range(num_exp):
        total_rew = 0
        observation = env.reset()
        for _ in range(num_iterations):  # 1000
            action = agent.act(observation, None, None) # np.random.randint(1, 3)
            observation, reward, done, info = env.step(action)
            total_rew += reward
            if done:
                break
        total_rewards.append(total_rew)

    print("random agent gets total reward: {}".format(np.mean(total_rewards)))
