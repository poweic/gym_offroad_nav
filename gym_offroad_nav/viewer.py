from gym_offroad_nav.rendering import ReferenceFrame

def assert_int(x, msg):
    assert int(x) == x, msg

class Viewer(object):
    def __init__(self, env):
        self.env = env
        self.viewport_scale = env.opts.viewport_scale

        assert_int(self.viewport_scale, "viewport_scale must be integer, not float")

        self.local_frame = ReferenceFrame(
            scale=self.viewport_scale / env.map.cell_size
        )

        self.viewer = None

    def initialized(self):
        return self.viewer is not None

    def render(self, **kwargs):
        if self.viewer is None:
            self._init_viewer()

        for sensor in self.env.sensors.itervalues():
            sensor.render()

        return self.viewer.render(**kwargs)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # All the rendering goes here...
    def _init_viewer(self):
        from gym_offroad_nav.rendering import Image, Viewer

        # Alias for width, height, and scaling. Note that the scaling factor
        # s is used only for rendering, so it won't affect any underlying
        # simulation. Just like zooming in/out the GUI and that's all.
        w, h, s = self.env.map.width, self.env.map.height, self.viewport_scale

        # Create viewer
        self.viewer = Viewer(width=w, height=h, scale=s)

        # Convert reward to uint8 image (by normalizing) and add as background
        self.viewer.add_geom(Image(
            img=self.env.map.rgb_map,
            center=(w/2, h/2), scale=s
        ))

        self.viewer.add_geom(self.local_frame)

        self.local_frame.transform.translation = (self.viewer.width/2., 0)

    def add(self, obj):
        self.local_frame.add_geom(obj)
