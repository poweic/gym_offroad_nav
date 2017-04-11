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

for i in range(1000):
    # print "========== RESET =========="
    env.reset()
    done = False

    total_return = 0

    while not np.any(done):
        action = env.env.sample_action()
        action[0] *= np.arange(len(action[0]))
        state, reward, done, _ = env.step(action.squeeze())
        total_return = total_return + reward.squeeze()
        print "total_return = ({:6.2f}, {:6.2f})".format(total_return[0], total_return[1])
        env.render()
        # cv2.waitKey(50)
