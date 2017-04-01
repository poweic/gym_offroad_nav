import cv2
import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def to_image(R, K=1, interpolation=cv2.INTER_NEAREST):
    R = normalize(R)
    R = cv2.resize(R, (R.shape[1] * K, R.shape[0] * K), interpolation=interpolation)[..., None]
    R = np.concatenate([R, R, R], axis=2)
    return R

def normalize(x):
    value_range = np.max(x) - np.min(x)
    if value_range != 0:
        x = (x - np.min(x)) / value_range * 255.
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def clip(x, minimum, maximum):
    return np.clip(x, minimum, maximum).astype(np.int32)

def get_options_from_tensorflow_flags():

    options = AttrDict({
        'field_of_view': 20,
        'min_mu_vf':  6. / 3.6,
        'max_mu_vf': 14. / 3.6,
        'min_mu_steer': -30 * np.pi / 180,
        'max_mu_steer': +30 * np.pi / 180,
        'timestep': 0.0025,
        'odom_noise_level': 0.02,
        'wheelbase': 2.0,
        'map_def': 'map0',
        'command_freq': 5,
        'n_agents_per_worker': 1,
        'viewport_scale': 5,
        'drift': False
    })

    try:
        import tensorflow as tf
        for key in options.keys():
            options[key] = getattr(tf.flags.FLAGS, key, options[key])
    except:
        pass

    return options
