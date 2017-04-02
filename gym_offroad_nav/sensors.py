import cv2
import abc
import numpy as np
from gym_offroad_nav.utils import to_image

RAD2DEG = 180. / np.pi

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
    def __init__(self, map, field_of_view, noise_level=0):
        super(FrontViewer, self).__init__(noise_level)

        self.map = map
        self.field_of_view = field_of_view
        self.noise_level = noise_level

        self.padded_rewards = pad_image(
            self.map.rewards,
            self.field_of_view
        )

        self.padded_rgb_map = pad_image(
            self.map.rgb_map,
            self.field_of_view
        )

        """
        cv2.imshow("padded_rewards", to_image(self.padded_rewards))
        cv2.waitKey(500)
        """

    def eval(self, state):
        
        cxs, cys, angles = self._get_cx_cy_angle(state)
        n_agents = len(angles)

        n_channels = 4

        height, width = self.padded_rewards.shape[:2]
        size = (width, height)
        fov = self.field_of_view
        print "fov = {}".format(fov)

        images = np.zeros((n_agents, fov, fov, n_channels), dtype=np.float32)
        for i, (cx, cy, angle) in enumerate(zip(cxs, cys, angles)):
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
            # print "[{}:{}, {}:{}]".format(cy-fov, cy, cx-fov/2, cx+fov/2)

            rotated_rewards = cv2.warpAffine(self.padded_rewards, M, size)
            rotated_rgb_map = cv2.warpAffine(self.padded_rgb_map, M, size)

            images[i, :, :, 0 ] = rotated_rewards[cy-fov:cy, cx-fov/2:cx+fov/2]
            images[i, :, :, 1:] = rotated_rgb_map[cy-fov:cy, cx-fov/2:cx+fov/2]

        # visualization (for debugging purpose)
        disp_img = np.zeros((2*fov, n_agents * fov, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            disp_img[:fov, i*fov:(i+1)*fov] += (img[..., 0:1] * 255).astype(np.uint8)
            disp_img[fov:, i*fov:(i+1)*fov] = img[..., -1:0:-1].astype(np.uint8)

        # cv2.imshow("reward", images[0, ..., 0])
        # cv2.imshow("rgb", images[0, ..., -1:0:-1].astype(np.uint8))
        cv2.imshow("front_view", disp_img)
        cv2.waitKey(5)

        return images

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
