import cv2
import abc
import math
import numpy as np
import time
from numba import jit
from gym_offroad_nav.utils import to_image
from gym_offroad_nav.lidar.lidar import c_lidar_mask
from gym import spaces

RAD2DEG = 180. / np.pi
FLOAT_MIN = np.finfo(np.float32).min
FLOAT_MAX = np.finfo(np.float32).max

# Every sensor is by definition noisy, by setting noise_level to 0 (default)
# the sensor is noise-free.
class SensorModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, noise_level=0):
        self.noise_level = noise_level

        # because we need to generate noise, we need a way to sync the random
        # number generator, so that we can playback.
        self.rng = np.random.RandomState()

    def seed(self, rng):
        self.rng = rng

    @abc.abstractmethod
    def eval(self, state):
        pass

    @abc.abstractmethod
    def render(self):
        pass

# odometry is also a sensor. The vehicle model is assumed to be perfect and 
# calculates how vehicle should move according to the laws of physics. We use
# odometry to measure how vehicle really move. If noise_level is 0, then this
# return a noise-free measurement, which is just the output of vehicle model.
class Odometry(SensorModel):
    def __init__(self, noise_level=0):
        super(Odometry, self).__init__()

    def eval(self, state):
        # Add some noise using state * (1 + noise) instead of state + noise
        noise = self.rng.rand(*state.shape) * self.noise_level
        return (state * (1 + noise)).T

    def render(self):
        # should render the trace of vehicle here
        pass

    def get_obs_space(self):
        return spaces.Box(low=FLOAT_MIN, high=FLOAT_MAX, shape=(6, 1))

def rotated_rect(img, (cx, cy), (width, height), angle, scale=1):
    pivot = (width/2, height)
    M = cv2.getRotationMatrix2D(pivot, angle, scale)
    M[:, 2] += (cx - width/2, cy - height)

    warped = cv2.warpAffine(
        img, M, (width, height), flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped

# @jit
def compute_obj_ixiy(obj_positions, state, fov, cell_size):
    theta = state[2]
    cos, sin = math.cos(theta), math.sin(theta)
    M = np.array([[cos, sin], [-sin, cos]])

    dxy = obj_positions - state[:2][..., None]
    dxy = M.dot(dxy)
    ixs, iys = (dxy / cell_size).astype(np.int)
    ixs, iys = ixs + fov/2, fov - 1 - iys

    return ixs, iys

# This sensor model return the front view of the vehicle as an image of shape
# fov x fov, where fov is the field of view (how far you see).
class FrontViewer(SensorModel):
    def __init__(self, map, field_of_view, noise_level=0):
        super(FrontViewer, self).__init__(noise_level)

        self.map = map
        self.field_of_view = field_of_view
        self.noise_level = noise_level

        # centralize the reward map, and rgb_map
        rewards = self.map.rewards[..., None].astype(np.float32)
        # self.reward_mean, self.reward_std = np.mean(rewards), np.std(rewards)
        # self.rewards = (rewards - self.reward_mean) / self.reward_std
        self.rewards = rewards
        self.rgb_map = self.map.rgb_map.astype(np.float32) / 127.5 - 1

        self.images = None

        self.timer = 0
        self.counter = 0

    def eval(self, state):
        
        cxs, cys, angles = self._get_cx_cy_angle(state)
        n_agents = len(angles)

        height, width = self.rewards.shape[:2]
        size = (width, height)
        fov = self.field_of_view

        n_channels = self.num_features()
        if self.images is None:
            self.images = np.zeros((n_agents, fov, fov, n_channels), dtype=np.float32, order='C')

        # W = self.get_waypoint_map(n_agents)

        obj_positions = np.concatenate([obj.position for obj in self.map.dynamic_objects], axis=-1)
        radius = int(obj.radius / self.map.cell_size)
        circle_opts = dict(color=(1, 1, 1), thickness=-1, radius=radius)

        for i, (cx, cy, angle, s) in enumerate(zip(cxs, cys, angles, state.T)):
            # features = np.concatenate([self.rewards, self.rgb_map, W[i]], axis=-1)
            # features = np.concatenate([self.rewards, W[i]], axis=-1)
            self.images[i] = rotated_rect(self.rewards, (cx, cy), (fov, fov), angle)[..., None]

            # iterate all dynamic_objects and draw on rotated_and_cropped image
            valids = [
                ((isinstance(obj.valid, bool) and obj.valid) or obj.valid[i])
                for obj in self.map.dynamic_objects
            ]
            ixs, iys = compute_obj_ixiy(obj_positions, s, fov, self.map.cell_size)

            for ix, iy, valid in zip(ixs, iys, valids):
                if valid:
                    cv2.circle(self.images[i], (ix, iy), **circle_opts)

        c_lidar_mask(self.images)

        return self.images

    def resize(self, images):
        n_agents, fov, fov, n_channels = images.shape
        out_shape = self.get_output_shape()[:2]

        images = images.transpose([1,2,3,0]).reshape(fov, fov, -1)

        images = cv2.resize(images, out_shape, interpolation=cv2.INTER_AREA)

        images = images.reshape(out_shape[0], out_shape[1], n_channels, -1)

        images = images.transpose([3, 0, 1, 2])

        return images

    def num_features(self):
        return 1

    def get_output_shape(self):
        m = int(self.field_of_view)
        return (m, m, self.num_features())

    def get_obs_space(self):
        return spaces.Box(low=FLOAT_MIN, high=FLOAT_MAX,
                          shape=self.get_output_shape())

    def render(self):

        if self.images is None:
            return

        n_agents, h, w = self.images.shape[:3]

        reward_min = -1 # np.min(self.map.class_id_to_rewards)
        reward_max = +1 # max(1, np.max(self.map.class_id_to_rewards))

        def unnormalize(x):
            # x = x * self.reward_std + self.reward_mean
            x = (x - reward_min) / (reward_max - reward_min)
            return x

        # visualization (for debugging purpose)
        disp_img = np.zeros((1*h, n_agents*w, 3), dtype=np.float32)
        for i, img in enumerate(self.images):
            s = slice(i*w, (i+1)*w)
            disp_img[0*h:1*h, s] += unnormalize(img[..., 0:1])
            # disp_img[1*h:2*h, s]  = (img[..., [3, 2, 1]] + 1) / 2
            # disp_img[2*h:3*h, s] += img[..., -1:]

        cv2.imshow("front_view", disp_img)
        cv2.waitKey(1)

    def get_waypoint_map(self, n_agents):

        m = -np.ones((n_agents,) + self.rewards.shape, dtype=np.float32)
        bounds = self.map.bounds
        fov = self.field_of_view

        for i in range(n_agents):
            for obj in self.map.dynamic_objects:
                # draw coin the this numpy ndarray, need to convert
                ix, iy = self.map.get_ixiy(*obj.position)
                x, y = ix[0] - bounds.x_min, bounds.y_max - 1 - iy[0]

                if (isinstance(obj.valid, bool) and obj.valid) or obj.valid[i]:
                    color = (1, 1, 1)
                else:
                    color = (-1, -1, -1)

                cv2.circle(m[i], (x, y), color=color, thickness=-1,
                           radius=int(obj.radius / self.map.cell_size))

        return m

    def _get_cx_cy_angle(self, state):
        x, y, theta = state[:3]

        ix, iy = self.map.get_ixiy(x, y)

        iix = np.clip(ix - self.map.bounds.x_min, 0, self.map.width - 1)
        iiy = np.clip(self.map.bounds.y_max - 1 - iy, 0, self.map.height - 1)

        fov = self.field_of_view
        cxs, cys = iix, iiy
        angles = theta * RAD2DEG

        return cxs, cys, angles


class Lidar(SensorModel):
    def __init__(self):
        raise NotImplementedError("Lidar not implemented yet")
