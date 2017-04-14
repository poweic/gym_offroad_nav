#!/usr/bin/python
import cv2
import gym
import gym_offroad_nav.envs
import numpy as np
import time

env = gym.make("OffRoadNav-v0")
# All the default_options are defined in gym_offroad_nav/envs/offroad_nav_env.py
# You can either change the default arguments or reconfigure the env here
# print env.env.default_options
env.env._configure({"map_def": "map5"})

# right now there's only spaces.Tuple in openai gym,
print env.observation_space

state = env.reset()
for key, obs in state.iteritems():
    print 'state["{}"]: shape = {}, dtype = {}'.format(key, obs.shape, obs.dtype)

np.set_printoptions(linewidth=1000,
    formatter={'float_kind': lambda x: "{:+7.2f}".format(x).replace('+', ' ')})

for i in range(1000):
    env.reset()
    done = False

    total_return = 0

    while not np.any(done):
        # sample actions for agents
        action = env.env.sample_action()

        # Change speed command so that it's more diverse (for debugging purpose)
        action[0] *= np.arange(len(action[0])).astype(np.float) / 10

        # step in the environment
        state, reward, done, _ = env.step(action.squeeze())

        # collect the reward
        total_return += reward
        print "total_return = {}".format(total_return)

        # refresh OpenGL renderer
        env.render()

        # (optinal) if you want to slow it down
        # cv2.waitKey(0)

    print "\n\n"
