#!/usr/bin/python
import cv2
import gym
import gym_offroad_nav.envs
import numpy as np
import time

env = gym.make("OffRoadNav-v0")
env.env._configure({
    "n_agents_per_worker": 16
})

for i in range(100):
    env.reset()
    done = False

    while not np.any(done):
        action = env.env.sample_action()
        state, reward, done, _ = env.step(action.squeeze())
        env.render()
