#!/usr/bin/python
import gym
import gym_offroad_nav.envs
import cv2

env = gym.make("OffRoadNav-v0")

for i in range(100):
    env.reset()
    done = False
    action = env.action_space.sample()
    while not done:
        # action = env.action_space.sample()
        state, reward, done, _ = env.step(action.squeeze())
        env.render()
        cv2.waitKey(10)

        """
        while cv2.waitKey(0) != 32:
            pass
        """
