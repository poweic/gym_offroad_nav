import yaml
import cv2
import numpy as np
from gym_offroad_nav.utils import AttrDict

metadata = AttrDict({
    'class_id_to_class_name': {
        0: 'void',
        1: 'smooth trail',
        2: 'low vegetation (traversable, drive cautiously, despite geometry)',
        3: 'slow down',
        4: 'water (not traversable, despite geometry)',
        5: 'obstacles (non traversable)',
        6: 'bushes (non traversable)',
        7: 'tree (non traversable)',
    },
    'class_id_to_rgb': {
        0: [255, 255, 255],
        1: [178, 176, 153],
        2: [128, 255, 0],
        3: [156, 76, 30],
        4: [255, 0, 128],
        5: [1, 88, 255],
        6: [0, 160, 0],
        7: [40, 80, 0]
    },
    'class_id_to_rewards': [
        0, 1, -1, 0.8, 0.4, -20, -10, -20
    ]
})

def save_yaml(fn, data):
    data = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in data.iteritems()
    }
    yaml.dump(data, open(fn, 'w'), width=1000)

def load_yaml(fn):
    data = yaml.load(open(fn, 'r'))
    return AttrDict({
        k: np.asarray(v) if isinstance(v, list) else v
        for k, v in data.iteritems()
    })

class OffRoadMap(object):
    def __init__(self, map_def_fn):

        # Load map definition from YAML file and store values as self attributes
        map_def = load_yaml(map_def_fn)
        for k, v in map_def.iteritems():
            setattr(self, k, v)

        self._init_boundary()

        self.rgb_map = self.colorize(self.map_structure)
        # cv2.imshow("rgb_map", self.rgb_map)
        # cv2.waitKey(500)

    def _init_boundary(self):

        self.height, self.width = self.map_structure.shape

        h, w = self.height, self.width
        cx, cy = 0, h / 2

        self.bounds = AttrDict(
            cx = cx,
            cy = cy,
            x_min = cx - w / 2,
            y_min = cy - h / 2,
            x_max = cx + w / 2,
            y_max = cy + h / 2,
        )

        """
        print "\33[33m(cx, cy) = ({}, {})".format(self.cx, self.cy)
        print "(x_min, x_max) = ({}, {}), (y_min, y_max) = ({}, {})".format(self.x_min, self.x_max, self.y_min, self.y_max)
        print "(height, width) = ({}, {})\33[0m".format(self.height, self.width)
        """

    def contains(self, x, y):
        b = self.bounds
        ix, iy = self.get_ixiy(x, y)
        inside = (ix >= b.x_min) & (ix <= b.x_max - 1) \
            & (iy >= b.y_min) & (iy <= b.y_max - 1)
        return inside

    def colorize(self, labels):
        img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        classes = np.unique(labels)
        for c in classes:
            img[labels == c] = metadata.class_id_to_rgb[c]
        return img

    def get_ixiy(self, x, y):
        ix = np.floor(x / self.cell_size).astype(np.int32)
        iy = np.floor(y / self.cell_size).astype(np.int32)
        return ix, iy

class DynamicObject(object):
    def __init__(self):
        pass

class Rewarder(object):
    def __init__(self, env):
        self.env = env

        self.static_rewarder = StaticRewarder(env.map)
        self.rewarders = []

    def eval(self, state):
        reward = self.static_rewarder.eval(state)
        for rewarder in self.rewarders:
            reward += rewarder.eval(state)
        return reward

class StaticRewarder(Rewarder):
    def __init__(self, map):

        self.map = map
        self.rewards = self.cvt_map_structure_to_rewards(map.map_structure)

    def cvt_map_structure_to_rewards(self, labels):
        rewards = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.float32)
        classes = np.unique(labels)
        for c in classes:
            rewards[labels == c] = metadata.class_id_to_rewards[c]
        return rewards

    def eval(self, state):
        x, y = state[:2]
        return self._bilinear_reward_lookup(x, y)

    def _bilinear_reward_lookup(self, x, y):
        ix, iy = self.map.get_ixiy(x, y)
        # print "(x, y) = ({}, {}), (ix, iy) = ({}, {})".format(x, y, ix, iy)

        # alias for self.map.bounds
        bounds = self.map.bounds

        x0 = np.clip(ix    , bounds.x_min, bounds.x_max - 1).astype(np.int32)
        y0 = np.clip(iy    , bounds.y_min, bounds.y_max - 1).astype(np.int32)
        x1 = np.clip(ix + 1, bounds.x_min, bounds.x_max - 1).astype(np.int32)
        y1 = np.clip(iy + 1, bounds.y_min, bounds.y_max - 1).astype(np.int32)

        f00 = self._get_reward(x0, y0)
        f01 = self._get_reward(x0, y1)
        f10 = self._get_reward(x1, y0)
        f11 = self._get_reward(x1, y1)

        xx = (x / self.map.cell_size - ix).astype(np.float32)
        yy = (y / self.map.cell_size - iy).astype(np.float32)

        w00 = (1.-xx) * (1.-yy)
        w01 = (   yy) * (1.-xx)
        w10 = (   xx) * (1.-yy)
        w11 = (   xx) * (   yy)

        r = f00*w00 + f01*w01 + f10*w10 + f11*w11
        return r.reshape(1, -1)

    def _get_reward(self, ix, iy):
        linear_idx = self._get_linear_idx(ix, iy)
        r = self.rewards.flatten()[linear_idx]
        return r

    def _get_linear_idx(self, ix, iy):
        bounds = self.map.bounds
        linear_idx = (bounds.y_max - 1 - iy) * self.map.width + (ix - bounds.x_min)
        return linear_idx

# Move the following classes to `sensors.py`
class SensorModel(object):
    def __init__(self):
        pass

class Lidar(SensorModel):
    def __init__(self):
        pass
