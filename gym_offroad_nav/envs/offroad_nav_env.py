import os
import gym
import cv2
import scipy.io
import numpy as np
from time import time
from gym import error, spaces, utils
from gym.utils import seeding

from gym_offroad_nav.utils import to_image, get_options_from_tensorflow_flags
from gym_offroad_nav.vehicle_model import VehicleModel
from gym_offroad_nav.vehicle_model_tf import VehicleModelGPU

RAD2DEG = 180. / np.pi

class OffRoadNavEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'], # 'video.frames_per_second': 30
    }

    def __init__(self):
        self.opts = get_options_from_tensorflow_flags()
        self.initialize()

    def _configure(self, opts):
        self.opts.update(opts)
        self.initialize()

    def initialize(self):

        self.fov = self.opts.field_of_view

        # action space = forward velocity + steering angle
        self.action_space = spaces.Box(low=np.array([self.opts.min_mu_vf, self.opts.min_mu_steer]), high=np.array([self.opts.max_mu_vf, self.opts.max_mu_steer]))
        self.dof = np.prod(self.action_space.shape)

        # Observation space = front view (image) + vehicle state (6-dim vector)
        float_min = np.finfo(np.float32).min
        float_max = np.finfo(np.float32).max
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=(self.fov, self.fov, 1)),
            spaces.Box(low=float_min, high=float_max, shape=(6, 1))
        ))

        # A matrix containing rewards, we need a constant version and 
        self.rewards = self.load_rewards()
        self.height, self.width = self.rewards.shape
        self.cx, self.cy = 0, self.height / 2
        self.x_min, self.x_max = self.cx - self.width  / 2, self.cx + self.width  / 2
        self.y_min, self.y_max = self.cy - self.height / 2, self.cy + self.height / 2

        """
        print "\33[33m(cx, cy) = ({}, {})".format(self.cx, self.cy)
        print "(x_min, x_max) = ({}, {}), (y_min, y_max) = ({}, {})".format(self.x_min, self.x_max, self.y_min, self.y_max)
        print "(height, width) = ({}, {})\33[0m".format(self.height, self.width)
        """
        
        self.cell_size = 0.5

        # self.vehicle_model_gpu = VehicleModelGPU(...)
        self.vehicle_model = VehicleModel(
            self.opts.timestep, self.opts.vehicle_model_noise_level,
            self.opts.wheelbase, self.opts.drift
        )

        self.K = 10

        self.state = None

        self.prev_action = np.zeros((2, 1))

        # Rendering
        self.viewer = None

        self.highlight = False

        self.rng = np.random.RandomState()

    def sample_action(self):
        return np.concatenate([
            self.action_space.sample()[:, None]
            for _ in range(self.opts.n_agents_per_worker)
        ], axis=1)

    def load_rewards(self):

        reward_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../data"
        reward_fn = "{}/{}.mat".format(reward_dir, self.opts.track)
        rewards = scipy.io.loadmat(reward_fn)["reward"].astype(np.float32)
        # rewards -= 100
        # rewards -= 15
        rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
        # rewards = (rewards - self.cell_size) * 2 # 128
        rewards = (rewards - 0.7) * 2
        rewards[rewards > 0] *= 10

        # rewards[rewards < 0.1] = -1

        return rewards

    def _seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        self.vehicle_model.seed(self.rng)
        return [seed]

    def _get_obs(self):
        return {
            "vehicle_state": self.state.copy().T,
            "front_view": self.get_front_view(self.state).copy()
        }

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
        action = action.reshape(self.dof, -1)
        n_sub_steps = int(1. / self.opts.command_freq / self.opts.timestep)
        for j in range(n_sub_steps):
            self.state = self.vehicle_model.predict(self.state, action)

        # Y forward, X lateral
        # ix = -20, -18, ...0, 1, 19, iy = 0, 1, ... 39
        x, y = self.state[:2]
        ix, iy = self.get_ixiy(x, y)
        done = (ix < self.x_min) | (ix > self.x_max - 1) | (iy < self.y_min) | (iy > self.y_max - 1)

        reward = self._bilinear_reward_lookup(x, y)

        # debug info
        info = {}

        self.prev_action = action.copy()

        return self._get_obs(), reward, done, info

    def get_linear_idx(self, ix, iy):
        linear_idx = (self.y_max - 1 - iy) * self.width + (ix - self.x_min)
        return linear_idx

    def _get_reward(self, ix, iy):
        linear_idx = self.get_linear_idx(ix, iy)
        r = self.rewards.flatten()[linear_idx]
        return r

    def get_ixiy(self, x, y, scale=1.):
        ix = np.floor(x * scale / self.cell_size).astype(np.int32)
        iy = np.floor(y * scale / self.cell_size).astype(np.int32)
        return ix, iy

    def _bilinear_reward_lookup(self, x, y):
        ix, iy = self.get_ixiy(x, y)
        # print "(x, y) = ({}, {}), (ix, iy) = ({}, {})".format(x, y, ix, iy)

        x0 = np.clip(ix    , self.x_min, self.x_max - 1).astype(np.int32)
        y0 = np.clip(iy    , self.y_min, self.y_max - 1).astype(np.int32)
        x1 = np.clip(ix + 1, self.x_min, self.x_max - 1).astype(np.int32)
        y1 = np.clip(iy + 1, self.y_min, self.y_max - 1).astype(np.int32)

        f00 = self._get_reward(x0, y0)
        f01 = self._get_reward(x0, y1)
        f10 = self._get_reward(x1, y0)
        f11 = self._get_reward(x1, y1)

        xx = (x / self.cell_size - ix).astype(np.float32)
        yy = (y / self.cell_size - iy).astype(np.float32)

        w00 = (1.-xx) * (1.-yy)
        w01 = (   yy) * (1.-xx)
        w10 = (   xx) * (1.-yy)
        w11 = (   xx) * (   yy)

        r = f00*w00 + f01*w01 + f10*w10 + f11*w11
        return r.reshape(1, -1)

    def get_initial_state(self):
        # state = np.array([+1, 1, -10 * np.pi / 180, 0, 0, 0])
        state = np.array([-32.5, 10.2, -10 * np.pi / 180, 0, 0, 0])

        # Reshape to compatiable format
        state = state.astype(np.float32).reshape(6, -1)

        # Add some noise to have diverse start points
        noise = self.rng.randn(6, self.opts.n_agents_per_worker).astype(np.float32) * 0.5
        noise[2, :] /= 2

        state = state + noise

        return state

    def _reset(self):
        if not hasattr(self, "padded_rewards"):
            fov = self.fov
            shape = (np.array(self.rewards.shape) + [fov * 2, fov * 2]).tolist()
            fill = np.min(self.rewards)
            self.padded_rewards = np.full(shape, fill, dtype=np.float32)
            self.padded_rewards[fov:-fov, fov:-fov] = self.rewards

        if not hasattr(self, "bR"):
            self.bR = to_image(self.debug_bilinear_R(), 1)

        self.disp_img = np.copy(self.bR)

        s0 = self.get_initial_state()
        self.vehicle_model.reset(s0)
        # self.vehicle_model_gpu.reset(s0)

        self.state = s0.copy()

        if self.viewer:
            for vehicle, state in zip(self.vehicles, self.state.T):
                vehicle.reset(state)

        return self._get_obs()

    def _init_viewer(self):
        from gym.envs.classic_control import rendering
        from gym_offroad_nav.rendering import Image, ReferenceFrame, Vehicle

        # Create viewer
        height, width = self.bR.shape[:2]
        self.viewer = rendering.Viewer(width=width, height=height)

        bg_img = Image(self.bR)
        bg_img.add_attr(rendering.Transform(translation=(width/2, height/2)))
        self.viewer.add_geom(bg_img)
    
    def _init_local_frame(self):
        from gym.envs.classic_control import rendering
        from gym_offroad_nav.rendering import Image, ReferenceFrame, Vehicle

        height, width = self.bR.shape[:2]

        scale = self.K / self.cell_size
        self.local_frame = ReferenceFrame(
            rendering.Transform(
                translation=(width/2., 0),
                scale=(scale, scale)
            )
        )

    def _init_vehicles(self):
        from gym.envs.classic_control import rendering
        from gym_offroad_nav.rendering import Image, ReferenceFrame, Vehicle

        self.vehicles = [
            Vehicle(keep_trace=True)
            for _ in range(self.opts.n_agents_per_worker)
        ]

        for vehicle in self.vehicles:
            self.local_frame.add_geom(vehicle)

    def debug_bilinear_R(self):
        X = np.linspace(self.x_min, self.x_max, num=self.width  * self.K) * self.cell_size
        Y = np.linspace(self.y_min, self.y_max, num=self.height * self.K) * self.cell_size

        xx, yy = np.meshgrid(X, Y)

        bR = self._bilinear_reward_lookup(xx, yy).reshape(xx.shape)

        # reverse Y-axis for image display
        bR = bR[::-1, :]

        return bR

    def get_front_view(self, state):
        x, y, theta = state[:3]

        ix, iy = self.get_ixiy(x, y)

        iix = np.clip(ix - self.x_min, 0, self.width - 1)
        iiy = np.clip(self.y_max - 1 - iy, 0, self.height - 1)

        fov = self.fov

        cxs, cys = iix + fov, iiy + fov

        angles = -theta * RAD2DEG

        img = np.zeros((len(angles), fov, fov), dtype=np.float32)
        for i, (cx, cy, angle) in enumerate(zip(cxs, cys, angles)):
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
            rotated = cv2.warpAffine(self.padded_rewards, M, self.padded_rewards.T.shape)
            # print "[{}:{}, {}:{}]".format(cy-fov, cy, cx-fov/2, cx+fov/2)
            img[i, :, :] = rotated[cy-fov:cy, cx-fov/2:cx+fov/2]

        return img[..., None]

    def get_vehicle_color(self, state):
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm as cmx

        # v^2 = v_x^2 + v_y^2
        v = np.sqrt(state[3] ** 2 + state[4] ** 2)

        # just rough guess of maximum velocity since we're using np.tanh to
        # squash it anyway
        vmax = self.action_space.high[0]
        v = (1 + np.tanh(v / vmax)) / 2

        scalarMap = cmx.ScalarMappable(
            norm=colors.Normalize(vmin=0, vmax=1),
            cmap=plt.get_cmap('hsv')
        )

        # HSV (1-v) / 3 ranges from green (v=0) to red (v=1)
        bgr = scalarMap.to_rgba((1-v)/3, bytes=True)[:3][::-1]
        bgr = (np.asscalar(bgr[0]), np.asscalar(bgr[1]), np.asscalar(bgr[2]))

        return bgr

    def _init_rendering(self):
        if self.viewer is None:
            self._init_viewer()
            self._init_local_frame()
            self._init_vehicles()
            self.viewer.add_geom(self.local_frame)

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
