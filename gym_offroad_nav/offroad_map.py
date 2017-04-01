import cv2
import yaml
import numpy as np
from gym_offroad_nav.utils import AttrDict, dirname
from gym_offroad_nav.interactable import OffRoadScene, Coin

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
    def __init__(self, map_def):

        # Load map definition from YAML file and store values as self attributes
        map_def = "{}/../maps/{}.yaml".format(dirname(__file__), map_def)
        map_def = load_yaml(map_def)
        for k, v in map_def.iteritems():
            setattr(self, k, v)

        self._init_boundary()

        self.rgb_map = self.colorize(self.map_structure)
        self.rewards = self.cvt_map_structure_to_rewards(self.map_structure)

        # TODO
        # Maybe we can use the idea of "context" to create static/dynamic object
        self.scene = OffRoadScene(map=self)
        self.static_objects = []
        self.dynamic_objects = []

        self.interactables = [self.scene] + self.static_objects + self.dynamic_objects

        # cv2.imshow("rgb_map", self.rgb_map)
        # cv2.waitKey(500)

    def colorize(self, labels):
        img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        classes = np.unique(labels)
        for c in classes:
            img[labels == c] = self.class_id_to_rgb[c]
        return img

    def cvt_map_structure_to_rewards(self, labels):
        rewards = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.float32)
        classes = np.unique(labels)
        for c in classes:
            rewards[labels == c] = self.class_id_to_rewards[c]
        return rewards

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

    def get_ixiy(self, x, y):
        ix = np.floor(x / self.cell_size).astype(np.int32)
        iy = np.floor(y / self.cell_size).astype(np.int32)
        return ix, iy

class Rewarder(object):
    def __init__(self, map):
        self.map = map

    # TODO This function to take an Agent class as input, and let agent interact
    # all objects in the environments, collect the result (in this case, the
    # reward), sum them up, and return the total reward.
    def eval(self, state):

        reward = 0
        for obj in self.map.interactables:
            reward += obj.react(state)
        return reward
