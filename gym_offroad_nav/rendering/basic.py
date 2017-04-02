from copy import deepcopy
import numpy as np
from collections import deque
from gym_offroad_nav.rendering.core import (
    import_pyglet, PolyLine, Color, Transform, Point, Geom, Viewer
)

class Image(Geom):

    def __init__(self, img, center=(0., 0.), scale=1.0):
        super(Image, self).__init__()
        self.attrs = []

        self.img = self.to_pyglet_image(img)

        self.height = self.img.height
        self.width = self.img.width
        self.scale = scale

        # center is default to the image center
        self.center = (
            -self.width  / 2 + center[0],
            -self.height / 2 + center[1]
        )

        self.add_attr(Color((1, 1, 1, 1)))
        self.flip = False

    def to_pyglet_image(self, img):

        import pyglet

        if type(img) == str:
            return pyglet.image.load(img)

        height, width, channel = img.shape

        if channel == 1:
            img = np.repeat(img, 3, axis=-1)

        if channel < 4:
            img = np.concatenate([img, np.ones((height, width, 1), dtype=np.uint8) * 255], axis=-1)

        image = pyglet.image.ImageData(
            width, height, 'RGBA', img.tobytes(),
            pitch= -width * 4
        )

        return image

    def render1(self):
        self.img.blit(
            self.center[0] * self.scale, self.center[1] * self.scale,
            width=self.width * self.scale, height=self.height * self.scale
        )

class ReferenceFrame(Geom):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=1.):
        super(ReferenceFrame, self).__init__()
        self.transform = Transform(
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
