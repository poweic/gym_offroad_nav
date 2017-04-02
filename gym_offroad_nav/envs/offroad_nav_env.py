import colored_traceback.always

import os
import gym
import cv2
import numpy as np
from time import time
from gym import error, spaces, utils
from gym.utils import seeding

from gym_offroad_nav.utils import get_options_from_TF_flags, AttrDict
from gym_offroad_nav.offroad_map import OffRoadMap, Rewarder
from gym_offroad_nav.sensors import Odometry, FrontViewer
from gym_offroad_nav.vehicle_model import VehicleModel
from gym_offroad_nav.vehicle_model_tf import VehicleModelGPU

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

    def initialize(self):

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

        self.state = None

        # action space = forward velocity + steering angle
        self.action_space = spaces.Box(
            low=np.array([self.opts.min_mu_vf, self.opts.min_mu_steer]),
            high=np.array([self.opts.max_mu_vf, self.opts.max_mu_steer])
        )
        self.dof = np.prod(self.action_space.shape)

        # Observation space = front view (image) + vehicle state (6-dim vector)
        fov = self.opts.field_of_view
        float_min = np.finfo(np.float32).min
        float_max = np.finfo(np.float32).max

        # TODO observation_space should be automatically deduced from sensors
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=(fov, fov, 1)),
            spaces.Box(low=float_min, high=float_max, shape=(6, 1))
        ))

        # Rendering
        self.viewer = None

        self.rng = np.random.RandomState()

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
        Tuple
            A 4-element tuple (state, reward, done, info)
        '''
        action = action.reshape(self.dof, self.opts.n_agents_per_worker)
        n_sub_steps = int(1. / self.opts.command_freq / self.opts.timestep)
        for j in range(n_sub_steps):
            self.state = self.vehicle_model.predict(self.state, action)

        # Y forward, X lateral
        x, y = self.state[:2]
        done = ~self.map.contains(x, y)

        reward = self.rewarder.eval(self.state)

        # debug info
        info = {}

        self.prev_action = action.copy()

        return self._get_obs(), reward, done, info

    def get_initial_state(self):
        # state = np.array([+1, 1, -10 * np.pi / 180, 0, 0, 0])
        state = np.array(self.map.initial_pose)

        # Reshape to compatiable format
        state = state.astype(np.float32).reshape(6, -1)

        # Add some noise to have diverse start points
        noise = self.rng.randn(6, self.opts.n_agents_per_worker).astype(np.float32) * 0.5
        noise[2, :] /= 2

        state = state + noise

        return state

    def _reset(self):
        s0 = self.get_initial_state()
        self.vehicle_model.reset(s0)
        # self.vehicle_model_gpu.reset(s0)

        self.state = s0.copy()

        if self.viewer:
            for vehicle, state in zip(self.vehicles, self.state.T):
                vehicle.reset(state)

            for obj in self.map.dynamic_objects:
                obj.reset()

        return self._get_obs()

    # All the rendering goes here...
    def _init_viewer(self):
        from gym_offroad_nav.rendering import Image, Viewer

        # Alias for width, height, and scaling. Note that the scaling factor
        # s is used only for rendering, so it won't affect any underlying
        # simulation. Just like zooming in/out the GUI and that's all.
        w, h, s = self.map.width, self.map.height, self.opts.viewport_scale
        assert int(s) == s, "viewport_scale must be integer, not float"

        # Create viewer
        self.viewer = Viewer(width=w, height=h, scale=s)

        # Convert reward to uint8 image (by normalizing) and add as background
        self.viewer.add_geom(Image(
            img=self.map.rgb_map,
            center=(w/2, h/2), scale=s
        ))
    
    def _init_local_frame(self):
        from gym_offroad_nav.rendering import ReferenceFrame

        self.local_frame = ReferenceFrame(
            translation=(self.viewer.width/2., 0),
            scale=self.opts.viewport_scale / self.map.cell_size
        )

    def _init_vehicles(self):
        from gym_offroad_nav.interactable import Vehicle

        self.vehicles = [
            Vehicle(keep_trace=True)
            for _ in range(self.opts.n_agents_per_worker)
        ]

        for vehicle in self.vehicles:
            self.local_frame.add_geom(vehicle)

    def _init_rendering(self):
        if self.viewer is None:
            self._init_viewer()
            self._init_local_frame()
            self._init_vehicles()
            self.viewer.add_geom(self.local_frame)

            for obj in self.map.dynamic_objects:
                self.local_frame.add_geom(obj)

    def _render(self, mode='human', close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        self._init_rendering()

        for vehicle, state in zip(self.vehicles, self.state.T):
            vehicle.set_pose(state)

        self.viewer.render()
