import cv2
import abc
import math
import time
import scipy.io
import numpy as np
from gym_offroad_nav.utils import to_image, Timer, dirname
from gym_offroad_nav.lidar.lidar import c_lidar_mask
from gym_offroad_nav.snapshot import memory_snapshot_decorate
from gym import spaces

# @memory_snapshot_decorate("tests/test_lidar/test_case_1.pkl")
def lidar_mask(images, threshold, random_seed):
    return c_lidar_mask(images, threshold, random_seed)

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

def rotated_rect(img, pivot, center, size, angle, scale=1):
    cx, cy = center
    width, height = size
    M = cv2.getRotationMatrix2D(pivot, angle, scale)
    M[:, 2] += (cx - pivot[0], cy - pivot[1])

    warped = cv2.warpAffine(
        img, M, (width, height), flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped

def compute_obj_ixiy(obj_positions, state, fov, pivot, cell_size):
    theta = state[2]
    cos, sin = math.cos(theta), math.sin(theta)
    M = np.array([[cos, sin], [-sin, cos]])

    dxy = obj_positions - state[:2][..., None]
    dxy = M.dot(dxy)
    ixs, iys = (dxy / cell_size).astype(np.int)
    ixs, iys = ixs + pivot[0], -iys + pivot[1]

    return ixs, iys

# This sensor model return the front view of the vehicle as an image of shape
# fov x fov, where fov is the field of view (how far you see).
class FrontViewer(SensorModel):
    def __init__(self, map, field_of_view, vehicle_position, noise_level=0):
        super(FrontViewer, self).__init__(noise_level)

        self.map = map
        self.field_of_view = field_of_view
        self.noise_level = noise_level

        self.masks = self.init_dropout_mask()

        fov = field_of_view
        if vehicle_position is "center":
            self.vehicle_position = (fov/2, fov - 1 - fov/2)
        elif vehicle_position is "bottom":
            self.vehicle_position = (fov/2, fov - 1 - 0)
        else:
            raise ValueError("vehicle position must be either center or bottom")

        #
        self.rewards = self.map.rewards[..., None].astype(np.float32)

        self.images = None

        # For classes that are not traversable, the rewards are actually the
        # porosity. Ex: bushes has a porosity of 0.4, so we store -0.4. We use
        # the minimum porosity * 0.95 as the threshold (take np.max because
        # they are all negative).
        self.pass_through_thres = np.max(
            self.map.class_id_to_rewards[~self.map.traversable]
        ) * 0.95

        self.timer = Timer("raycasting")
        self.counter = 0

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

        masks = (masks == 0)

        return masks

    def eval(self, state):
        
        cxs, cys, angles = self._get_cx_cy_angle(state)
        n_agents = len(angles)

        height, width = self.rewards.shape[:2]
        size = (width, height)
        fov = self.field_of_view

        n_channels = self.num_features()
        if self.images is None:
            self.images = np.zeros((n_agents, fov, fov, n_channels), dtype=np.float32, order='C')

        obj_positions = np.concatenate([obj.position for obj in self.map.dynamic_objects], axis=-1)
        radius = int(obj.radius / self.map.cell_size)
        circle_opts = dict(color=(1, 1, 1), thickness=-1, radius=radius)
        vpos = self.vehicle_position

        cxy_angles = zip(cxs, cys, angles, state.T)

        for i, (cx, cy, angle, s) in enumerate(cxy_angles):
            self.images[i] = rotated_rect(self.rewards, vpos, (cx, cy), (fov, fov), angle)[..., None]

            # Draw the vehicle itself as a dot
            # self.images[i, vpos[1], vpos[0]] = 1

        self.timer.tic()
        random_seed = self.rng.randint(low=2, high=np.iinfo(np.uint32).max)
        lidar_mask(self.images, self.pass_through_thres, random_seed)

        n_masks = len(self.masks)
        for img in self.images:
            idx = self.rng.randint(low=0, high=n_masks)
            mask = self.masks[idx]
            img[mask] = -1

        for i, (cx, cy, angle, s) in enumerate(cxy_angles):
            # iterate all dynamic_objects and draw on rotated_and_cropped image
            valids = [
                ((isinstance(obj.valid, bool) and obj.valid) or obj.valid[i])
                for obj in self.map.dynamic_objects
            ]
            ixs, iys = compute_obj_ixiy(obj_positions, s, fov, vpos, self.map.cell_size)

            for ix, iy, valid in zip(ixs, iys, valids):
                if valid:
                    cv2.circle(self.images[i], (ix, iy), **circle_opts)

        self.timer.toc()

        return self.images

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

        reward_min = -1
        reward_max = +1

        def unnormalize(x):
            x = (x - reward_min) / (reward_max - reward_min)
            return x

        # visualization (for debugging purpose)
        disp_img = np.zeros((h, n_agents*w, 3), dtype=np.float32)
        for i, img in enumerate(self.images):
            s = slice(i*w, (i+1)*w)
            disp_img[0*h:1*h, s] += unnormalize(img[..., 0:1])

        # disp_img = cv2.resize(disp_img, (disp_img.shape[1]*4, disp_img.shape[0]*4), interpolation=cv2.INTER_NEAREST)
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
