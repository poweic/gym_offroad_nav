#!/usr/bin/python
import gym
import gym_offroad_nav.envs

env = gym.make("OffRoadNav-v0")
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action.squeeze())
    env.render()
