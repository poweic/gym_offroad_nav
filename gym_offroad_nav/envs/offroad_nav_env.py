import colored_traceback.always

import os
import gym
import cv2
import numpy as np
from time import time
from gym import error, spaces, utils
from gym.utils import seeding
from attrdict import AttrDict

from gym_offroad_nav.utils import get_options_from_TF_flags, Timer, get_speed
from gym_offroad_nav.offroad_map import OffRoadMap, Rewarder
from gym_offroad_nav.sensors import Odometry, FrontViewer
from gym_offroad_nav.vehicle_model.numpy_impl import VehicleModel
from gym_offroad_nav.viewer import Viewer
# from gym_offroad_nav.vehicle_model.tf_impl import VehicleModelGPU
# from gym_offroad_nav.trajectory_following import TrajectoryFitter

from lifelines import KaplanMeierFitter
from collections import deque

DEG2RAD = np.pi / 180.

class OffRoadNavEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'], # 'video.frames_per_second': 30
    }

    default_options = {
        'field_of_view': 128,
        'min_mu_vf': -14. / 3.6,
        'max_mu_vf': +14. / 3.6,
        'min_mu_steer': -30 * DEG2RAD,
        'max_mu_steer': +30 * DEG2RAD,
        'timestep': 0.025,
        'odom_noise_level': 0.02,
        'vehicle_model_noise_level': 0.02,
        'initial_pose_noise': [0, 0, 5 * DEG2RAD, 0, 0, 5 * DEG2RAD],
        'avg_state_window': 30,
        'wheelbase': 2.0,
        'map_def': 'map9',
        'command_freq': 5,
        'vehicle_position': 'bottom', # can be either at center, or bottom
        'KMF_window': 100,
        'n_agents_per_worker': 32,
        'viewport_scale': 0.5,
        'discount_factor': 0.99,
        'max_steps': 10000,
        'drift': False
    }

    def __init__(self):
        # self.max_l2_norm = 0
        self.opts = AttrDict(self.default_options)
        self.opts.update(get_options_from_TF_flags(self.opts.keys()))
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
        self.map = OffRoadMap(self.opts.map_def, self.opts.n_agents_per_worker)

        # A matrix containing rewards, we need a constant version and 
        self.rewarder = Rewarder(self.map)

        # create sensor models, now we just use a possibly noisy odometry and
        # the simplest front view sensor
        self.sensors = {
            'vehicle_state': Odometry(self.opts.odom_noise_level),
            'front_view': FrontViewer(
                self.map, self.opts.field_of_view, self.opts.vehicle_position
            )
        }

        """
        self.vehicle_model_gpu = VehicleModelGPU(
            self.opts.timestep, self.opts.vehicle_model_noise_level,
            self.opts.wheelbase, self.opts.drift
        )
        """
        self.vehicle_model = VehicleModel(
            self.opts.timestep, self.opts.vehicle_model_noise_level,
            self.opts.wheelbase, self.opts.drift
        )

        self.state = np.zeros((6, self.opts.n_agents_per_worker))

        # action space = forward velocity + steering angle
        self.action_space = spaces.Box(
            low=np.array([self.opts.min_mu_vf, self.opts.min_mu_steer]),
            high=np.array([self.opts.max_mu_vf, self.opts.max_mu_steer])
        )
        self.dof = np.prod(self.action_space.shape)

        self.n_sub_steps = int(1. / self.opts.command_freq / self.opts.timestep)

        # observation_space is automatically deduced from sensors
        self.observation_space = spaces.Tuple((
            self.sensors['front_view'].get_obs_space(),
            self.sensors['vehicle_state'].get_obs_space(),
        ))

        # TrajectoryFitter
        # use default session if exists, create on otherwise
        """
        import tensorflow as tf
        self.sess = tf.get_default_session() or tf.Session()
        self.traj_fitter = TrajectoryFitter(
            sess=self.sess, vehicle_model=self.vehicle_model,
            margin=0.5, n_steps=self.n_sub_steps
        )
        """

        # Rendering
        self.viewer = Viewer(self)

        self.init_agents()

        self.rng = np.random.RandomState()

        self.kmf = KaplanMeierFitter()
        self.last_N_distances_traveled = deque(maxlen=self.opts.KMF_window)
        self.last_N_crashed = deque(maxlen=self.opts.KMF_window)

        self.total_episodes = 0

        self.timer = AttrDict(
            vehicle_model=Timer("vehicle_model"),
            get_obs=Timer("get_obs"),
            others=Timer("contains + reward")
        )

        self.initialized = True

    def fit(self):
        waypoints = self.map.waypoints[2::2, None, :]
        print waypoints.shape

        s_target = np.repeat(waypoints, [self.opts.n_agents_per_worker], axis=1)
        s_target = np.pad(s_target, [[0, 0], [0, 0], [0, 4]], 'constant')
        s_target = np.concatenate([self.state.T[None], s_target])
        print s_target.shape

        loss, actions = self.traj_fitter.fit(s_target)

        return actions

    def sample_action(self):
        return np.concatenate([
            self.action_space.sample()[:, None]
            for _ in range(self.opts.n_agents_per_worker)
        ], axis=1)

    def _seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        self.map.seed(self.rng)
        self.vehicle_model.seed(self.rng)
        for sensor in self.sensors.itervalues():
            sensor.seed(self.rng)
        return [seed]

    def _get_obs(self):
        return AttrDict({
            k: sensor.eval(self.state).copy() for k, sensor in self.sensors.iteritems()
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
        new_state = self.state.copy()
        # new_state_gpu = self.state.copy()

        new_state, rewards, distances_traveled = self.vehicle_model.predict(
            new_state, action, self.n_sub_steps, self.map
        )
        self.distances_traveled += distances_traveled
        self.timer.vehicle_model.toc()

        """
        new_state_gpu = self.vehicle_model_gpu.predict(new_state_gpu, action, self.n_sub_steps, self.sess)
        diff = new_state - new_state_gpu
        l2norm = np.linalg.norm(diff.flatten())
        self.max_l2_norm = max(l2norm, self.max_l2_norm)
        print "L2 norm = {:12.7e}, allclose = {}, max_l2_norm = {:12.7e}".format(
            l2norm, np.allclose(new_state, new_state_gpu), self.max_l2_norm
        )
        """

        self.timer.others.tic()
        # See if the car is in tree. If the speed is too high, then it's crashed
        cls = self.map.get_class(new_state)
        impact = ~self.map.traversable[cls]
        vel = get_speed(new_state)
        crashed = impact & (vel > 1.)
        crash_penalty = crashed * self.map.crash_penalty
        impact_penalty = vel * impact * self.map.impact_penalty

        # collect statistics when episode ends (i.e. any of the vehicle crashed)
        if np.any(crashed):
            self.total_episodes += 1
            self.last_N_distances_traveled.append(self.distances_traveled)
            self.last_N_crashed.append(crashed)
            self.kmf_summarize()

        info = AttrDict()

        # compute reward based on new_state
        info.reward = rewards + self.rewarder.eval(new_state) - crash_penalty - impact_penalty
        self.total_reward += info.reward
        reward = np.mean(info.reward)

        # if new position is in the tree, then use old one & set velocity = 0
        new_state[0:3, impact] = self.state[0:3, impact]
        new_state[3:6, impact] = 0
        self.vehicle_model.reset(new_state, impact)

        self.state[:] = new_state[:]

        # Determine whether it's done
        x, y = self.state[:2]
        info.done = ~self.map.contains(x, y) | crashed
        done = np.any(info.done)

        self.timer.others.toc()

        self.timer.get_obs.tic()
        self.obs = self._get_obs()
        self.timer.get_obs.toc()

        self.prev_action = action.copy()
        self.steps += 1

        if self.steps >= self.opts.max_steps:
            done = True

        return self.obs, reward, done, info

    def get_initial_state(self):
        # state is 6-dimensional vector (x, y, theta, x', y', theta'),
        # where Y is forward, X is lateral
        state = np.array(self.map.initial_pose)

        # Reshape to compatiable format
        state = state.reshape(6, -1)

        # Generate some noise to have diverse start points
        noise = self.rng.randn(6, self.opts.n_agents_per_worker)
        scale = np.array(self.opts.initial_pose_noise)[..., None]
        noise = noise * scale

        # Add noise to state
        state = state + noise

        return state

    def init_agents(self):
        from gym_offroad_nav.interactable import Vehicle

        self.vehicles = [
            Vehicle(
                pose=self.state.T[i], keep_trace=False, draw_horizon=False,
                time_per_step=1. / self.opts.command_freq,
                discount_factor=self.opts.discount_factor
            )
            for i in range(self.opts.n_agents_per_worker)
        ]

    def kmf_summarize(self):
        distances = np.array(self.last_N_distances_traveled)
        event_observed = np.array(self.last_N_crashed)
        self.kmf.fit(distances.reshape(-1), event_observed=event_observed.reshape(-1))

        if self.total_episodes % self.opts.KMF_window == 0:
            print "\33[92mKMF's MDTF = {}, naive MDTF = {}\33[0m".format(
                self.kmf.median_, np.mean(np.sum(distances, axis=-1)))

    def _reset(self):
        self.map.reset()
        s0 = self.get_initial_state()
        self.vehicle_model.reset(s0)
        # self.vehicle_model_gpu.reset(s0)

        self.distances_traveled = 0
        self.crashes = 0

        self.state[:] = s0[:]
        for vehicle in self.vehicles:
            vehicle.reset()

        self.total_reward = 0
        self.steps = 0

        self.viewer.clear()
        self.add_objects_to_viewer()

        return self._get_obs()

    def add_objects_to_viewer(self):

        for vehicle in self.vehicles:
            self.viewer.add(vehicle)

        for obj in self.map.dynamic_objects:
            self.viewer.add(obj)

    def _render(self, mode='human', close=False):

        if close:
            self.viewer.close()
            return

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
