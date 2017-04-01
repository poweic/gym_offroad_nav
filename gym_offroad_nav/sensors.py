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

def pad_image(img, padding, fill_value=None):
    shape = (np.array(img.shape) + [padding * 2, padding * 2]).tolist()
    if fill_value is None:
        fill_value = np.min(img)
    padded_x = np.full(shape, fill_value, dtype=img.dtype)
    padded_x[padding:-padding, padding:-padding] = img
    return padded_x

# This sensor model return the front view of the vehicle as an image of shape
# fov x fov, where fov is the field of view (how far you see).
class FrontViewer(SensorModel):
    def __init__(self, map, rewarder, field_of_view, noise_level=0):
        super(FrontViewer, self).__init__(noise_level)

        self.map = map
        self.rewarder = rewarder
        self.field_of_view = field_of_view

        self.padded_rewards = pad_image(
            self.rewarder.static_rewarder.rewards,
            self.field_of_view
        )

        """
        self.padded_rgb_map = pad_image(
            self.map.rgb_map,
            self.field_of_view
        )
        """

        """
        cv2.imshow("padded_rewards", to_image(self.padded_rewards))
        cv2.waitKey(500)
        """

    def get_padded_rewards(self):
        return self.padded_rewards

    def get_padded_rgb_map(self):
        return self.padded_rgb_map

    def eval(self, state):
        x, y, theta = state[:3]

        ix, iy = self.map.get_ixiy(x, y)

        iix = np.clip(ix - self.map.bounds.x_min, 0, self.map.width - 1)
        iiy = np.clip(self.map.bounds.y_max - 1 - iy, 0, self.map.height - 1)

        fov = self.field_of_view

        cxs, cys = iix + fov, iiy + fov

        angles = -theta * RAD2DEG

        padded_rewards = self.get_padded_rewards()

        img = np.zeros((len(angles), fov, fov), dtype=np.float32)
        for i, (cx, cy, angle) in enumerate(zip(cxs, cys, angles)):
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
            rotated = cv2.warpAffine(padded_rewards, M, padded_rewards.T.shape)
            # print "[{}:{}, {}:{}]".format(cy-fov, cy, cx-fov/2, cx+fov/2)
            img[i, :, :] = rotated[cy-fov:cy, cx-fov/2:cx+fov/2]

        # cv2.imshow("front_view", img[0])
        # cv2.waitKey(5)

        return img[..., None]

class Lidar(SensorModel):
    def __init__(self):
        raise NotImplementedError("Lidar not implemented yet")
