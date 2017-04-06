import os
import cv2
import time
import numpy as np
from collections import deque

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def to_image(R, K=1, interpolation=cv2.INTER_NEAREST):
    R = normalize(R)
    R = cv2.resize(R, (R.shape[1] * K, R.shape[0] * K), interpolation=interpolation)[..., None]
    R = np.concatenate([R, R, R], axis=2)
    return R

def dirname(fn):
    return os.path.dirname(os.path.realpath(fn))

def normalize(x):
    value_range = np.max(x) - np.min(x)
    if value_range != 0:
        x = (x - np.min(x)) / value_range * 255.
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def clip(x, minimum, maximum):
    return np.clip(x, minimum, maximum).astype(np.int32)

# Get options from TensorFlow FLAGS, use default values if not provided
def get_options_from_TF_flags():

    options = AttrDict({
        'field_of_view': 64,
        'min_mu_vf':  6. / 3.6,
        'max_mu_vf': 14. / 3.6,
        'min_mu_steer': -30 * np.pi / 180,
        'max_mu_steer': +30 * np.pi / 180,
        'timestep': 0.025,
        'odom_noise_level': 0.02,
        'wheelbase': 2.0,
        'map_def': 'map2',
        'command_freq': 5,
        'n_agents_per_worker': 16,
        'viewport_scale': 4,
        'drift': False
    })

    try:
        import tensorflow as tf
        for key in options.keys():
            options[key] = getattr(tf.flags.FLAGS, key, options[key])
    except Exception as e:
        print e
        pass

    return options


class Timer(object):
    def __init__(self, message, maxlen=1000):
        self.timer = deque(maxlen=maxlen)
        self.counter = 0
        self.message = message

    def tic(self):
        self.t = time.time()

    def toc(self):
        self.timer.append(time.time() - self.t)

        self.counter += 1
        if self.counter % self.timer.maxlen == 0:
            self.counter = 0

            import tensorflow as tf
            tf.logging.set_verbosity(tf.logging.INFO)
            tf.logging.info("average time of {} = {:.2f} ms".format(
                self.message, np.mean(self.timer) * 1000
            ))
