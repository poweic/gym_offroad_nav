#!/usr/bin/env python
import cv2
import gym
import gym_offroad_nav.envs
import numpy as np
import time
try:
    # import my logger from https://bitbucket.org/castacks/deep_rl
    import drl.logger
except:
    # but it's optional, just ignore this
    pass
from gym_offroad_nav.joystick import JoystickController

env = gym.make("OffRoadNav-v0")
# All the default_options are defined in gym_offroad_nav/envs/offroad_nav_env.py
# You can either change the default arguments or reconfigure the env here
# print env.env.default_options
# env.env._configure({"map_def": "map5"})

# right now there's only spaces.Tuple in openai gym,
print env.observation_space

state = env.reset()
for key, obs in state.iteritems():
    print 'state["{}"]: shape = {}, dtype = {}'.format(key, obs.shape, obs.dtype)

np.set_printoptions(linewidth=1000,
    formatter={'float_kind': lambda x: "{:+7.2f}".format(x).replace('+', ' ')})

total_return = 0

def callback(controls):
    global total_return

    if controls.buttons[0]:
        env.reset()
        total_return = 0

    action_vf    = -(controls.pitch - 512.) / 512. * 6
    # action_vf    = (controls.throttle- 128.) / 128. * 6

    # action_steer =  (controls.yaw - 128.) / 128. * (30 / 180. * np.pi)
    action_steer =  (controls.roll - 512.) / 512. * (30 / 180. * np.pi)
    # print action_vf, action_steer
    action = np.array([action_vf, action_steer])

    # step in the environment
    state, reward, done, _ = env.step(action.squeeze())

    # collect the reward
    total_return += reward
    print "total_return = {} [done = {}]".format(total_return, done)

    # refresh OpenGL renderer
    env.render()

joystick = JoystickController(callback)
joystick.start()

for i in range(1000):
    env.reset()
    done = False

    total_return = 0

    '''
    actions = env.env.fit()
    cv2.waitKey(0)
    '''

    i = 0
    while not np.any(done):
        '''
        if i >= len(actions): break
        action = actions[i]
        '''
        # sample actions for agents
        action = env.env.sample_action()

        # Change speed command so that it's more diverse (for debugging purpose)
        # action[0] *= np.arange(len(action[0])).astype(np.float) / 10
        action[0] = 5

        # step in the environment
        state, reward, done, _ = env.step(action.squeeze())

        # collect the reward
        total_return += reward
        # print "total_return = {}".format(total_return)

        # refresh OpenGL renderer
        env.render()

        # (optinal) if you want to slow it down
        # cv2.waitKey(0)
        i += 1

    # print "\n"
