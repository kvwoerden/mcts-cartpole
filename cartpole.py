import argparse
import sys

import gym
from gym import wrappers, logger

import numpy as np
import time
import copy
import random
import statistics

from collections import defaultdict

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from mcts import UctSearch

import random


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class RolloutAgent(object):
    def __init__(self, n_actions, rollouts, environment_name):
        self.agenttype = 'rollout'
        self.n_actions = n_actions
        self.rollouts = rollouts
        self.environment_name = environment_name

    def act(self, state, debug=False):
        results = defaultdict(list)
        for action in range(self.n_actions):
            action_env = gym.make(self.environment_name)
            action_env.reset()
            if debug:
                pass
                # print("### 401", action_env.get_state())
                # print("### 501", action_env.state.__class__)
                # action_env.state = np.array(state)
                # action_env.set_state(np.array(state))
            # action_env.state = tuple(state)
            action_env.unwrapped.state = np.array(state)
            if debug:
                # print("### 401.1", action_env.get_state())
                action_env.step(0)
                action_env.step(0)
                action_env.step(0)
                action_env.step(0)
                # print("### 402", action_env.get_state())
                input()
            # action_env.state = copy.deepcopy(state)
            if debug:
                print("### 301 action_env_state: ", action_env.state)
            after_act_state, reward, done_state, info = action_env.step(action)
            if debug:
                # print("### 302 action_env_state: ", action_env.get_state())
                print("### 303 action_env_state: ", action_env.state)
                input()
            for _ in range(self.rollouts):
                if debug:
                    print("### 303 action_env_state: ", action_env.state)
                    print("### 303.1 action_env_state: ",
                          action_env.state.__class__)
                done = done_state
                rollout_env = gym.make(self.environment_name)
                rollout_env.reset()
                # rollout_env.set_state(np.array(action_env.get_state()))
                # rollout_env.state = tuple(action_env.state)
                rollout_env.unwrapped.state = np.array(action_env.state)
                if debug:
                    print("### 304 rollout_env_state: ", rollout_env.state)
                    input()
                sum_reward = 0
                # print("### 201")
                # rollout_env.render()
                # input()
                # done = False
                while not done:
                    # print("### 101 \n")
                    # rollout_env.render()
                    # input()
                    random_action = random.randrange(0, self.n_actions)
                    after_state, reward, done, info = rollout_env.step(
                        random_action)
                    # print("### 101", state, sum_reward)
                    sum_reward += reward
                    if debug:
                        print("### 200 sum_reward", sum_reward)
                        print("### 600 ", rollout_env.state.__class__)
                        input()
                    # print("### 102", state, sum_reward)
                    # input()
                # print("### 102", reward)
                results[action].append(sum_reward)
        if debug:
            print(sorted(results[0]))
            input()
        # averages = {k: statistics.mean(v) for k, v in results.items()}
        maxima = {k: max(v) for k, v in results.items()}
        medians = {k: statistics.median(v) for k, v in results.items()}
        print("### 001", medians)
        # print("### 002", {k: len(v) for k, v in results.items()})
        # print("### 003", results[1])
        # best = max(averages, key=averages.get)
        # best = max(maxima, key=maxima.get)
        best = max(medians, key=medians.get)
        print("### 004 choosing", best)
        print('### 005 len(results)', len(results[0]), len(results[1]))
        print('### 006 min max', min(results[0]), max(results[0]))
        print('### 007 min max', min(results[1]), max(results[1]))
        return best


class CartPoleAgent(object):
    """CartPole agent!"""

    def __init__(self, action_space, observation_space, epsilon):
        self.action_space = action_space
        self.observation_space = observation_space
        self.epsilon = epsilon
        self.cart_pos_bins = np.arange(-2.4 - 0.96, 2.4 + 0.96, 4.8/5)
        self.cart_vel_bins = np.arange(-1e30, 1e30, 1e30)
        self.pole_angle_bins = np.arange(-14, 14, 2)
        self.pole_vel_bins = np.arange(-1e30, 1e30, 1e30)
        self.Q = np.random.normal(0, 0.01, (self.cart_pos_bins.size, self.cart_vel_bins.size,
                                            self.pole_angle_bins.size, self.pole_angle_bins.size, self.action_space.n))
        self.Q[(0, self.cart_pos_bins.size - 1), :, :, :, :] = 0
        self.Q[:, :, (0, self.pole_angle_bins.size - 1), :, :] = 0


class MCTSAgent(object):
    def __init__(self, time_budget, debug, environment):
        self.agenttype = 'mcts'
        self.time_budget = time_budget
        self.debug = debug
        self.environment = environment

    def act(self, params, n_actions, node, all_nodes, C_p, lookahead_target):
        return UctSearch(params, n_actions, self.environment, self.time_budget, node=node, all_nodes=all_nodes, C_p=C_p, lookahead_target=lookahead_target)


if __name__ == '__main__':
    SEED = 28
    random.seed(SEED)
    parser = argparse.ArgumentParser(description=None)
    ENVIRONMENT = 'CartPole-v0'
    parser.add_argument('env_id', nargs='?', default=ENVIRONMENT,
                        help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    env = gym.make(args.env_id)
    # env.metadata['video.frames_per_second'] = 24
    # print(env.metadata)
    # input()
    # rec = VideoRecorder(env, path='./video/output01.mp4')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    # outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    # env.seed(0)
    env.seed(1)
    TIME_BUDGET = 10
    ITERATION_BUDGET = 4000
    LOOKAHEAD_TARGET = 50
    DEBUG = False
    agent = MCTSAgent(ITERATION_BUDGET, DEBUG, ENVIRONMENT)

    episode_count = 1
    TIMESTR = time.strftime("%Y%m%d-%H%M%S")

    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        rec = VideoRecorder(env, path='./video/output' +
                            TIMESTR + '-' + str(i) + '.mp4')
        try:
            sum_reward = 0
            node = None
            all_nodes = []
            C_p = 20
            while True:
                print("################")
                env.render()
                rec.capture_frame()
                action, node, all_nodes, C_p = agent.act(
                    env.state, n_actions=env.action_space.n, node=node, all_nodes=all_nodes, C_p=C_p, lookahead_target=LOOKAHEAD_TARGET)
                ob, reward, done, _ = env.step(action)
                print("### 101 observed state: ", ob)
                print("### 201 C_p: ", C_p)
                sum_reward += reward
                print("### 008 sum_reward: ", sum_reward)
                if done:
                    rec.close()
                    break
        except KeyboardInterrupt as e:
            rec.close()
            env.close()
            raise e

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk

    env.close()
