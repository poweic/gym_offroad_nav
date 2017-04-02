import abc
import numpy as np
from copy import deepcopy
from collections import deque
from gym_offroad_nav.utils import AttrDict, clip
from gym_offroad_nav.rendering import PolyLine, Transform, Geom, Color, Point

class Interactable(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, map):
        self.map = map

    @abc.abstractmethod
    def react(self, state):
        pass

class DynamicObject(Interactable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(DynamicObject, self).__init__(**kwargs)

    def react(self, state):
        return 0

class StaticObject(Interactable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(StaticObject, self).__init__(**kwargs)

    @abc.abstractmethod
    def react(self, state):
        pass

class Coin(Interactable):

    def __init__(self, position, radius, reward, **kwargs):
        super(OffRoadScene, self).__init__(**kwargs)

        self.position = position
        self.radius = radius
        self.reward = reward

        # One should only call this when rendering
        # self.transform = rendering.Transform()

    def react(self, state):
        return 1

class Vehicle(Geom):
    def __init__(self, size=2., keep_trace=False, max_trace_length=100):
        super(Vehicle, self).__init__()

        self.size = size
        self.keep_trace = keep_trace

        # pose of vehicle (translation + rotation)
        self.transform = Transform()

        # vertices of vehicles
        h, w = size, size / 2
        r, l, t, b = w/2, -w/2, h/2, -h/2
        vertices = [(l,b), (l,t), (r,t), (r,b), (0, 0), (l, b), (r, b)]
        self.polyline = PolyLine(vertices, close=False)
        self.polyline.attrs = [Color((1, 0, 0, 1)), self.transform]

        # trace of vehicle (historical poses)
        self.trace = deque(maxlen=max_trace_length)

        self.reset()

    def set_pose(self, pose):
        # pose is a 3-dimensional vector (x, y, theta)
        x, y, theta = pose[:3]
        self.transform.set_translation(x, y)
        self.transform.set_rotation(theta)

        if self.keep_trace:
            p = Point()
            p.attrs = [Color((1, 0, 0, 1)), deepcopy(self.transform)]
            self.trace.append(p)

    def reset(self, pose=[0., 0., 0.]):
        self.trace.clear()
        self.set_pose(pose)

    def render1(self):
        self.polyline.render()
        for p in self.trace:
            p.render()

class OffRoadScene(Interactable):

    def __init__(self, **kwargs):
        super(OffRoadScene, self).__init__(**kwargs)

    def react(self, state):
        x, y = state[:2]
        return self._bilinear_reward_lookup(x, y)

    def _bilinear_reward_lookup(self, x, y):
        ix, iy = self.map.get_ixiy(x, y)

        # alias for self.map.bounds
        bounds = self.map.bounds

        x0 = clip(ix    , bounds.x_min, bounds.x_max - 1)
        y0 = clip(iy    , bounds.y_min, bounds.y_max - 1)
        x1 = clip(ix + 1, bounds.x_min, bounds.x_max - 1)
        y1 = clip(iy + 1, bounds.y_min, bounds.y_max - 1)

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
        bounds = self.map.bounds
        r = self.map.rewards[bounds.y_max - 1 - iy, ix - bounds.x_min]

        # this is legacy code, make sure I didn't break it
        linear_idx = (bounds.y_max - 1 - iy) * self.map.width + (ix - bounds.x_min)
        assert np.all(r == self.map.rewards.flatten()[linear_idx])

        return r
