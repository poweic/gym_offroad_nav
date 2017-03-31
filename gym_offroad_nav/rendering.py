from copy import deepcopy
import pyglet
import numpy as np
from collections import deque
from gym.envs.classic_control import rendering

Viewer = rendering.Viewer

class Image(rendering.Geom):

    def __init__(self, img, center=(0., 0.), scale=1.0):
        super(Image, self).__init__()
        self.attrs = []

        if type(img) == str:
            self.img = pyglet.image.load(img)
        else:
            self.img = self.to_pyglet_image(img)

        self.height = self.img.height
        self.width = self.img.width
        self.scale = scale

        self.center_trans = rendering.Transform(translation=center)
        self.add_attr(self.center_trans)

        self.add_attr(rendering.Color((1, 1, 1, 1)))
        self.flip = False

    def to_pyglet_image(self, ndarray):

        height, width, channel = ndarray.shape

        if channel == 1:
            ndarray = np.repeat(ndarray, 3, axis=-1)

        if channel < 4:
            ndarray = np.concatenate([ndarray, np.ones((height, width, 1), dtype=np.uint8) * 255], axis=-1)

        image = pyglet.image.ImageData(
            width, height, 'RGBA', ndarray.tobytes(),
            pitch= -width * 4
        )

        return image

    def render1(self):
        self.img.blit(
            -self.width/2 * self.scale, -self.height/2 * self.scale,
            width=self.width * self.scale, height=self.height * self.scale
        )

class ReferenceFrame(rendering.Geom):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=1.):
        super(ReferenceFrame, self).__init__()
        self.transform = rendering.Transform(
            translation=translation, rotation=rotation, scale=(scale, scale)
        )
        self.add_attr(self.transform)
        self.geoms = []
        self.onetime_geoms = []

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render1(self):
        for g in self.geoms:
            g.render()
        for g in self.onetime_geoms:
            g.render()
        self.onetime_geoms = []

class Vehicle(rendering.Geom):
    def __init__(self, size=2., keep_trace=False, max_trace_length=100):
        super(Vehicle, self).__init__()

        self.size = size
        self.keep_trace = keep_trace

        # pose of vehicle (translation + rotation)
        self.transform = rendering.Transform()

        # vertices of vehicles
        h, w = size, size / 2
        r, l, t, b = w/2, -w/2, h/2, -h/2
        vertices = [(l,b), (l,t), (r,t), (r,b), (0, 0), (l, b), (r, b)]
        self.polyline = rendering.PolyLine(vertices, close=False)
        self.polyline.attrs = [rendering.Color((1, 0, 0, 1)), self.transform]

        # trace of vehicle (historical poses)
        self.trace = deque(maxlen=max_trace_length)

        self.reset()

    def set_pose(self, pose):
        # pose is a 3-dimensional vector (x, y, theta)
        x, y, theta = pose[:3]
        self.transform.set_translation(x, y)
        self.transform.set_rotation(theta)

        if self.keep_trace:
            p = rendering.Point()
            p.attrs = [rendering.Color((1, 0, 0, 1)), deepcopy(self.transform)]
            self.trace.append(p)

    def reset(self, pose=[0., 0., 0.]):
        self.trace.clear()
        self.set_pose(pose)

    def render1(self):
        self.polyline.render()
        for p in self.trace:
            p.render()
