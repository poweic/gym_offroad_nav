import abc
import cv2
import numpy as np
from copy import deepcopy
from collections import deque
from gym_offroad_nav.utils import AttrDict, clip, dirname, get_speed
from gym_offroad_nav import rendering

class Interactable(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, map, *args, **kwargs):
        self.map = map

    @abc.abstractmethod
    def react(self, state):
        pass

class DynamicObject(Interactable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(DynamicObject, self).__init__(**kwargs)

    def react(self, state):
        return 0

class StaticObject(Interactable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(StaticObject, self).__init__(**kwargs)

    @abc.abstractmethod
    def react(self, state):
        pass

class Coin(Interactable, rendering.Geom):

    def __init__(self, position, radius=2., reward=10., **kwargs):
        Interactable.__init__(self, **kwargs)
        rendering.Geom.__init__(self, **kwargs)

        self.position = np.array([position]).T
        self.radius = float(radius)
        self.reward = float(reward)

        self.transform = rendering.Transform(translation=position)
        color = rendering.Color((247./255, 223./255, 56./255, 0.9))
        self.coin = rendering.Image(dirname(__file__) + "/../assets/coin.png", scale=0.015)
        self.coin.attrs = [color, self.transform]

        self.coin_radius = rendering.make_circle(radius, filled=False)
        self.coin_radius.attrs = [color, self.transform]

        self.reset()

    def react(self, state):
        distance = np.linalg.norm(state[:2] - self.position, axis=0)
        inside = (distance <= self.radius)
        r = self.reward * (inside & self.valid)
        self.valid &= ~inside
        return r

    def reset(self):
        self.valid = True

    def render1(self):
        if not np.any(self.valid):
            return
        self.coin.render()
        self.coin_radius.render()

class Vehicle(rendering.Geom):
    def __init__(self, pose, size=2., keep_trace=False, draw_horizon=False,
                 max_trace_length=100, time_per_step=1., discount_factor=0.99):
        super(Vehicle, self).__init__()

        self.size = size
        self.keep_trace = keep_trace
        self.draw_horizon = draw_horizon
        self.max_trace_length = max_trace_length
        self.discount_factor = discount_factor
        self.time_per_step = time_per_step

        # pose of vehicle (translation + rotation)
        self.transform = rendering.Transform2D(pose=pose)

        # vertices of vehicles
        # TODO use cell size to determine how large we draw on canvas
        h, w = size, size / 2
        r, l, t, b = w/2, -w/2, h/2, -h/2
        vertices = [(l,b), (l,t), (r,t), (r,b), (0, 0), (l, b), (r, b)]
        self.polyline = rendering.PolyLine(vertices, close=False)
        self.polyline.attrs = [rendering.Color((1, 0, 0, 1)), self.transform]

        # draw horizon (based on discount_factor) as a circle
        self.k_steps_5_percent = np.log(0.05) / np.log(discount_factor)

        # trace of vehicle (historical poses)
        self.trace = deque(maxlen=max_trace_length)

        self.reset()

    def __setattr__(self, name, value):
        if name == "pose":
            raise AttributeError("Pose is immutable, please call set_pose()")
        else:
            return super(Vehicle, self).__setattr__(name, value)

    '''
    def set_pose(self, pose):
        # pose is a 6-dimensional vector (x, y, theta, x', y', theta')
        self.transform.set_pose(pose)
    '''

    def reset(self, pose=[0., 0., 0., 0., 0., 0.]):
        self.trace.clear()

    def get_horizon(self):

        speed = np.linalg.norm(self.transform.pose[3:5])
        radius = self.k_steps_5_percent * speed * self.time_per_step
        horizon = rendering.make_circle(radius, filled=False)
        horizon.attrs = [rendering.Color((0.3, 0.77, 1, 1)), self.transform]

        return horizon

    def render1(self):
        if self.keep_trace:
            p = rendering.Point()
            p.attrs = [rendering.Color((1, 0, 0, 1)), deepcopy(self.transform)]
            self.trace.append(p)

        self.polyline.render()

        if self.draw_horizon:
            self.get_horizon().render()

        for p in self.trace:
            p.render()

'''
class OffRoadScene(Interactable):

    def __init__(self, *args, **kwargs):
        super(OffRoadScene, self).__init__(**kwargs)

        self.impact_penalty = self.map.impact_penalty

    def react(self, state):

        x, y = state[:2]

        classes = self.map.get_class(x, y)
        impact = (classes >= 5)

        vel = get_speed(state)
        impact_penalty = self.impact_penalty * impact * vel

        return - impact_penalty
'''
