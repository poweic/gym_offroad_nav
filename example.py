#!/usr/bin/python
import cv2
import gym
import gym_offroad_nav.envs
import numpy as np
import time

env = gym.make("OffRoadNav-v0")
# env.env._configure({"n_agents_per_worker": 2})

for i in range(100):
    env.reset()
    done = False

    while not np.any(done):
        action = env.env.sample_action()
        state, reward, done, _ = env.step(action.squeeze())
        print "reward = ({:8.0f}, {:8.0f})".format(reward[0, 0], reward[0, 1])
        env.render()
        # cv2.waitKey(0)
