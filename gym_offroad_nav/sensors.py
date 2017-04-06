import cv2
import abc
import numpy as np
import time
from gym_offroad_nav.utils import to_image
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

def pad_image(img, pad, fill_value=None):
    if img.ndim == 2:
        img = img[..., None]

    shape = np.asarray(img.shape) + [pad * 2, pad * 2, 0]
    if fill_value is None:
        fill_value = np.min(img)
    padded_x = np.full(shape, fill_value, dtype=img.dtype)
    padded_x[pad:-pad, pad:-pad] = img
    return padded_x

# This sensor model return the front view of the vehicle as an image of shape
# fov x fov, where fov is the field of view (how far you see).
class FrontViewer(SensorModel):
    def __init__(self, map, field_of_view, downsample, noise_level=0):
        super(FrontViewer, self).__init__(noise_level)

        self.map = map
        self.field_of_view = field_of_view
        self.downsample = float(downsample)
        self.noise_level = noise_level

        self.padded_rewards = pad_image(
            self.map.rewards,
            self.field_of_view
        )

        self.padded_rgb_map = pad_image(
            self.map.rgb_map,
            self.field_of_view
        )

        self.images = None

        self.timer = 0
        self.counter = 0

    def eval(self, state):
        
        cxs, cys, angles = self._get_cx_cy_angle(state)
        n_agents = len(angles)

        R = self.padded_rewards
        C = self.padded_rgb_map
        W = self._get_padded_waypoint_map()

        height, width = R.shape[:2]
        size = (width, height)
        fov = self.field_of_view

        n_channels = 5
        images = np.zeros((n_agents, fov, fov, n_channels), dtype=np.float32)

        # self.timer -= time.time()
        for i, (cx, cy, angle) in enumerate(zip(cxs, cys, angles)):
            # print "[{}:{}, {}:{}]".format(cy-fov, cy, cx-fov/2, cx+fov/2)
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1)

            sx = slice(cx-fov/2, cx+fov/2)
            sy = slice(cy-fov, cy)

            images[i, ..., 0  ] = cv2.warpAffine(R, M, size)[sy, sx]
            images[i, ..., 1:4] = cv2.warpAffine(C, M, size)[sy, sx]
            images[i, ..., 4  ] = cv2.warpAffine(W, M, size)[sy, sx]
        # self.timer += time.time()
        # self.counter += 1

        # downsample the image
        images = self.resize(images)

        self.images = images

        return images

    def resize(self, images):
        n_agents, fov, fov, n_channels = images.shape
        out_shape = self.get_output_shape()[:2]

        images = images.transpose([1,2,3,0]).reshape(fov, fov, -1)

        images = cv2.resize(images, out_shape, interpolation=cv2.INTER_AREA)

        images = images.reshape(out_shape[0], out_shape[1], n_channels, -1)

        images = images.transpose([3, 0, 1, 2])

        return images

    def get_output_shape(self):
        m = int(self.field_of_view / self.downsample)
        return (m, m, 5)

    def get_obs_space(self):
        return spaces.Box(low=FLOAT_MIN, high=FLOAT_MAX,
                          shape=self.get_output_shape())

    def render(self):

        if self.images is None:
            return

        n_agents, h, w = self.images.shape[:3]

        # visualization (for debugging purpose)
        disp_img = np.zeros((3*h, n_agents*w, 3), dtype=np.uint8)
        for i, img in enumerate(self.images):
            s = slice(i*w, (i+1)*w)
            disp_img[0*h:1*h, s] += (img[..., 0:1] * 255).astype(np.uint8)
            disp_img[1*h:2*h, s]  = (img[..., [3, 2, 1]]).astype(np.uint8)
            disp_img[2*h:3*h, s] += (img[..., 4:5] * 255).astype(np.uint8)

        cv2.imshow("front_view", disp_img)
        cv2.waitKey(1)

    def _get_padded_waypoint_map(self):

        if hasattr(self, 'padded_waypoint_map'):
            return self.padded_waypoint_map

        m = np.zeros_like(self.padded_rewards)
        bounds = self.map.bounds
        fov = self.field_of_view

        for obj in self.map.dynamic_objects:

            # TODO check whether the waypoint is still valid, but this means
            # we need to create separate waypoint map for each agent ...
            # if not obj.valid: continue

            # draw coin the this numpy ndarray, need to convert
            ix, iy = self.map.get_ixiy(*obj.position)
            x, y = ix - bounds.x_min, bounds.y_max - 1 - iy
            cv2.circle(m, (x+fov, y+fov), color=(1, 1, 1), thickness=-1,
                       radius=int(obj.radius / self.map.cell_size))

        self.padded_waypoint_map = m

        return m

    def _get_cx_cy_angle(self, state):
        x, y, theta = state[:3]

        ix, iy = self.map.get_ixiy(x, y)

        iix = np.clip(ix - self.map.bounds.x_min, 0, self.map.width - 1)
        iiy = np.clip(self.map.bounds.y_max - 1 - iy, 0, self.map.height - 1)

        fov = self.field_of_view
        cxs, cys = iix + fov, iiy + fov
        angles = -theta * RAD2DEG

        return cxs, cys, angles


class Lidar(SensorModel):
    def __init__(self):
        raise NotImplementedError("Lidar not implemented yet")
