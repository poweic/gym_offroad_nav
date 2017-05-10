import cv2
import abc
import math
import time
import scipy.io
import numpy as np
from gym_offroad_nav.utils import to_image, Timer, dirname, rescale_image
from gym_offroad_nav.lidar.lidar import c_lidar_mask
from gym_offroad_nav.snapshot import memory_snapshot_decorate
from gym import spaces

# @memory_snapshot_decorate("tests/test_lidar/test_case_1.pkl")
'''
def lidar_mask(images, threshold, random_seed):
    return c_lidar_mask(images, threshold, random_seed)
'''

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
        measurement = (state * (1 + noise)).T
        return measurement.astype(np.float32)

    def render(self):
        # should render the trace of vehicle here
        pass

    def get_obs_space(self):
        return spaces.Box(low=FLOAT_MIN, high=FLOAT_MAX, shape=(6, 1))

# convert [-1, 1] to [0, 255]
def rescale2uint8(x):
    x = (x + 1.) / 2. * 255.
    if isinstance(x, np.ndarray):
        return x.astype(np.uint8)
    else:
        return int(x)

# This sensor model return the front view of the vehicle as an image of shape
# fov x fov, where fov is the field of view (how far you see).
class FrontViewer(SensorModel):
    def __init__(self, map, field_of_view, vehicle_position, noise_level=0):
        super(FrontViewer, self).__init__(noise_level)

        self.map = map
        self.field_of_view = field_of_view
        self.noise_level = noise_level

        self.masks = self.init_dropout_mask()
        self.scale = 1.

        fov = field_of_view
        if vehicle_position is "center":
            self.vehicle_position = (fov/2, fov - 1 - fov/2)
        elif vehicle_position is "bottom":
            self.vehicle_position = (fov/2, fov - 1 - 0)
        else:
            raise ValueError("vehicle position must be either center or bottom")

        #
        self.rewards = self.init_reward_map()

        self.images = None

        # For classes that are not traversable, the rewards are actually the
        # porosity. Ex: bushes has a porosity of 0.4, so we store -0.4. We use
        # the minimum porosity * 0.95 as the threshold (take np.max because
        # they are all negative).
        self.pass_through_thres = rescale2uint8(np.max(
            self.map.class_id_to_rewards[~self.map.traversable]
        ) * 0.95)

        self.counter = 0

    def init_reward_map(self):
        rewards = self.map.rewards[..., None]

        # Make sure all rewards are within the range [-1, +1]
        assert -1 <= np.min(rewards) and np.max(rewards) <= 1

        rewards = rescale2uint8(rewards)

        return rewards

    def init_dropout_mask(self):

        current_dir = dirname(__file__)
        masks = scipy.io.loadmat(current_dir + "/masks.mat")['masks']

        N, H, W = masks.shape

        fov = self.field_of_view

        if fov != H:
            size = (fov, fov)
            masks = np.vstack([
                cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)[None]
                for mask in masks
            ])

        masks = masks.astype(np.uint8, order='C')

        return masks

    def eval(self, state):
        
        cxs, cys, angles = self._get_cx_cy_angle(state)
        n_agents = len(angles)

        height, width = self.rewards.shape[:2]
        size = (width, height)
        fov = self.field_of_view

        n_channels = self.num_features()
        if self.images is None:
            self.images = np.zeros((n_agents, fov, fov, n_channels), dtype=np.uint8, order='C')

        vpos = self.vehicle_position

        # =====================================================================
        centers = np.ascontiguousarray(np.array([cxs, cys], dtype=np.int32).T)
        angles = angles.astype(np.float32)
        states = np.ascontiguousarray(state.T)

        obj_positions = np.ascontiguousarray(
            np.concatenate([obj.position for obj in self.map.dynamic_objects], axis=-1).T
        )
        valids = self.valids(n_agents)
        radius = int(obj.radius / self.map.cell_size)

        random_seed = self.rng.randint(low=2, high=np.iinfo(np.uint32).max)

        c_lidar_mask(
            self.rewards, self.masks, centers, angles, vpos, self.scale,
            self.images, obj_positions, valids, states, self.map.cell_size, radius,
            self.pass_through_thres, random_seed
        )
        # =====================================================================

        '''
        for i in range(n_agents):
            # Draw the vehicle itself as a dot
            self.images[i, vpos[1], vpos[0]] = 1
        '''

        return self.images

    def valids(self, n_agents):

        valids = [[
            ((isinstance(obj.valid, bool) and obj.valid) or obj.valid[i])
            for obj in self.map.dynamic_objects
        ] for i in range(n_agents)]

        valids = np.array(valids, dtype=np.uint8, order='C')

        return valids

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

        # visualization (for debugging purpose)
        disp_img = np.concatenate([img for img in self.images], axis=1)

        # disp_img = rescale_image(disp_img, 4)
        cv2.imshow("front_view", disp_img)
        cv2.waitKey(1)

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
