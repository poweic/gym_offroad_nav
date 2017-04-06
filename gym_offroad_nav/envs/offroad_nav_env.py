import colored_traceback.always

import os
import gym
import cv2
import numpy as np
from time import time
from gym import error, spaces, utils
from gym.utils import seeding

from gym_offroad_nav.utils import get_options_from_TF_flags, AttrDict, Timer
from gym_offroad_nav.offroad_map import OffRoadMap, Rewarder
from gym_offroad_nav.sensors import Odometry, FrontViewer
from gym_offroad_nav.vehicle_model import VehicleModel
from gym_offroad_nav.vehicle_model_tf import VehicleModelGPU
from gym_offroad_nav.viewer import Viewer

class OffRoadNavEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'], # 'video.frames_per_second': 30
    }

    def __init__(self):
        self.opts = get_options_from_TF_flags()
        self.initialize()

    def _configure(self, opts):
        self.opts.update(opts)
        self.initialize()

    def __setattr__(self, name, value):
        if name == "state" and self.initialized:
            raise AttributeError("state is immutable once it's initialized.")
        else:
            return super(OffRoadNavEnv, self).__setattr__(name, value)

    def initialize(self):
        self.initialized = False

        # Load map definition from YAML file from configuration file
        self.map = OffRoadMap(self.opts.map_def)

        # A matrix containing rewards, we need a constant version and 
        self.rewarder = Rewarder(self.map)

        # create sensor models, now we just use a possibly noisy odometry and
        # the simplest front view sensor
        self.sensors = {
            'vehicle_state': Odometry(self.opts.odom_noise_level),
            'front_view': FrontViewer(self.map, self.opts.field_of_view)
        }

        # self.vehicle_model_gpu = VehicleModelGPU(...)
        self.vehicle_model = VehicleModel(
            self.opts.timestep, self.opts.wheelbase, self.opts.drift
        )

        self.state = np.zeros((6, self.opts.n_agents_per_worker), dtype=np.float32)

        # action space = forward velocity + steering angle
        self.action_space = spaces.Box(
            low=np.array([self.opts.min_mu_vf, self.opts.min_mu_steer]),
            high=np.array([self.opts.max_mu_vf, self.opts.max_mu_steer])
        )
        self.dof = np.prod(self.action_space.shape)

        # observation_space is automatically deduced from sensors
        self.observation_space = spaces.Tuple((
            self.sensors['front_view'].get_obs_space(),
            self.sensors['vehicle_state'].get_obs_space(),
            # spaces.Box(low=0, high=255, shape=(fov, fov, 5)),
            # spaces.Box(low=float_min, high=float_max, shape=(6, 1))
        ))

        print self.observation_space

        # Rendering
        self.viewer = Viewer(self)

        self.init_agents()

        for obj in self.map.dynamic_objects:
            self.viewer.add(obj)

        self.rng = np.random.RandomState()

        self.initialized = True

        self.timer = AttrDict(
            vehicle_model=Timer("vehicle_model"),
            get_obs=Timer("get_obs"),
            others=Timer("contains + reward")
        )

    def sample_action(self):
        return np.concatenate([
            self.action_space.sample()[:, None]
            for _ in range(self.opts.n_agents_per_worker)
        ], axis=1)

    def _seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        for sensor in self.sensors.itervalues():
            sensor.seed(self.rng)
        return [seed]

    def _get_obs(self):
        return AttrDict({
            k: sensor.eval(self.state) for k, sensor in self.sensors.iteritems()
        })

    def _step(self, action):
        ''' Take one step in the environment
        state is the vehicle state, not the full MDP state including history.

        Parameters
        ----------
        action : Numpy array
            The control input for vehicle. [v_forward, yaw_rate]

        Returns
        -------
        A 4-element tuple (state, reward, done, info)
        '''
        self.timer.vehicle_model.tic()
        action = action.reshape(self.dof, self.opts.n_agents_per_worker)
        n_sub_steps = int(1. / self.opts.command_freq / self.opts.timestep)
        state = self.state.copy()
        for j in range(n_sub_steps):
            state = self.vehicle_model.predict(state, action)
        self.state[:] = state[:]
        self.timer.vehicle_model.toc()

        # compute reward and determine whether it's done
        self.timer.others.tic()
        reward = self.rewarder.eval(self.state)
        self.total_reward += reward

        # Y forward, X lateral
        x, y = self.state[:2]
        done = ~self.map.contains(x, y)
        done |= (self.total_reward < self.map.minimum_score).squeeze()

        self.timer.others.toc()

        # debug info
        info = {}

        self.timer.get_obs.tic()
        self.obs = self._get_obs()
        self.timer.get_obs.toc()

        self.prev_action = action.copy()

        return self.obs, reward, done, info

    def get_initial_state(self):
        state = np.array(self.map.initial_pose)

        # Reshape to compatiable format
        state = state.astype(np.float32).reshape(6, -1)

        # Add some noise to have diverse start points
        noise = self.rng.randn(6, self.opts.n_agents_per_worker).astype(np.float32) * 0.5
        noise[2, :] /= 10

        state = state + noise

        return state

    def init_agents(self):
        from gym_offroad_nav.interactable import Vehicle

        self.vehicles = [
            Vehicle(pose=self.state.T[i], keep_trace=True)
            for i in range(self.opts.n_agents_per_worker)
        ]

        for vehicle in self.vehicles:
            self.viewer.add(vehicle)

    def _reset(self):
        s0 = self.get_initial_state()
        self.vehicle_model.reset(s0)
        # self.vehicle_model_gpu.reset(s0)

        self.state[:] = s0[:]
        self.total_reward = 0

        if self.viewer.initialized():
            for obj in self.map.dynamic_objects:
                obj.reset()

        return self._get_obs()

    def _render(self, mode='human', close=False):

        if close:
            self.viewer.close()
            return

        self.viewer.render()
