import os
import cv2
import time
import numpy as np
from collections import deque
from attrdict import AttrDict

def to_image(R, K=1, interpolation=cv2.INTER_NEAREST):
    R = normalize(R)
    R = cv2.resize(R, (R.shape[1] * K, R.shape[0] * K), interpolation=interpolation)[..., None]
    R = np.concatenate([R, R, R], axis=2)
    return R

def dirname(fn):
    return os.path.dirname(os.path.realpath(fn))

def get_speed(state):
    return np.sqrt(state[3] ** 2 + state[4] ** 2)

def get_position(state):
    return np.sqrt(state[0] ** 2 + state[1] ** 2)

def normalize(x):
    value_range = np.max(x) - np.min(x)
    if value_range != 0:
        x = (x - np.min(x)) / value_range * 255.
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def clip(x, minimum, maximum):
    return np.clip(x, minimum, maximum).astype(np.int32)

# Get options from TensorFlow FLAGS, use default values if not provided
def get_options_from_TF_flags(keys):
    options = AttrDict()
    try:
        import tensorflow as tf
        for key in keys:
            if hasattr(tf.flags.FLAGS, key):
                options[key] = getattr(tf.flags.FLAGS, key)
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

            try:
                import tensorflow as tf
                tf.logging.set_verbosity(tf.logging.INFO)
                tf.logging.info("average time of {} = {:.2f} ms".format(
                    self.message, np.mean(self.timer) * 1000
                ))
            except:
                pass
