import math
import numpy as np

RAD2DEG = 57.29577951308232

def import_pyglet():
    if 'pyglet' in globals():
        return

    global pyglet
    import pyglet.gl
    for k in dir(pyglet.gl):
        globals()[k] = pyglet.gl.__dict__[k]

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

class Viewer(object):
    def __init__(self, width, height, display=None, scale=1.0):
        width = int(width * scale)
        height = int(height * scale)

        import_pyglet()
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scalex),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):
    def __init__(self, *arg, **kwargs):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_alpha(self, a):
        self._color.vec4 = self._color.vec4[:3] + (a,)
    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

class Transform2D(Attr):
    def __init__(self, pose):
        self.pose = pose
    def enable(self):
        glPushMatrix()
        glTranslatef(self.pose[0], self.pose[1], 0)
        glRotatef(RAD2DEG * self.pose[2], 0, 0, 1.0)
    def disable(self):
        glPopMatrix()
    def set_pose(self, pose):
        self.pose[:] = pose[:]
    def set_translation(self, newx, newy):
        self.pose[0] = float(newx)
        self.pose[1] = float(newy)
    def set_rotation(self, new):
        self.pose[2] = float(new)

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)

class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
    def render1(self):
        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()

def make_circle(radius=10, res=20, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

def make_polyline(v):
    return PolyLine(v, False)

class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()

class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

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

"""
TODO: maybe I should link this library with ROS
class Grid(object):
    def __init__(self, center, ):

        self.attrs = []

        label = pyglet.text.Label(
            'Hello, world', font_name='Times New Roman', font_size=36, x=x, y=y,
            anchor_x='center', anchor_y='center', color=(255, 255, 255, 255)
        )

    def render1(self):
"""
