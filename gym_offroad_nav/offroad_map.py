import cv2
import yaml
import PIL.Image
import numpy as np
from gym_offroad_nav.utils import AttrDict, dirname
from gym_offroad_nav.interactable import OffRoadScene, Coin
from gym_offroad_nav.global_planner import GlobalPlanner

def get_palette_color(img):
    print np.frombuffer(img.palette.palette, np.uint8).reshape(-1, 3)

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

def load_rewards(track):
    import scipy.io
    fn = "{}/../data/{}.mat".format(dirname(__file__), track)
    rewards = scipy.io.loadmat(fn)['reward'].astype(np.float32)
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    # rewards = (rewards - self.cell_size) * 2 # 128
    rewards = (rewards - 0.7) * 2
    rewards[rewards > 0] *= 10

    return rewards

class OffRoadMap(object):
    def __init__(self, map_def, n_agents_per_worker):

        # Load map definition from YAML file and store values as self attributes
        self.load_map_def(map_def)
        self.n_agents_per_worker = n_agents_per_worker

        self._init_boundary()

        self.rgb_map = self.colorize(self.map_structure)
        self.rewards = self.cvt_map_structure_to_rewards(self.map_structure)
        # self.rewards = load_rewards("big_track")
        print self.rewards.shape

        sigma = 3
        # blur the reward map to get a more continuous (smoother) reward
        self.blurred_rewards = np.ascontiguousarray(
            cv2.GaussianBlur(self.rewards, (sigma, sigma), 0))

        # Random random generator for global planner to randomly sample paths
        self.rng = np.random.RandomState()
        self.global_planner = GlobalPlanner()

        # TODO
        # Maybe we can use the idea of "context" to create static/dynamic object
        # instead of passing map to every constructor.
        self.scene = OffRoadScene(map=self)
        self.static_objects = []

        self.reset()


    def seed(self, rng):
        self.rng = rng
        self.global_planner.seed(self.rng)

    def reset(self):
        # TODO "reset" is really a bad name, change it plz ...

        self.waypoints, self.initial_pose = self.create_waypoints()

        radius = self.waypoint.radius
        reward = self.waypoint.reward
        goal_r = self.waypoint.goal_reward

        # downsample the waypoints by 10 so that coins won't be too crowded.
        # Also, skip the first waypoint, since it's the start point
        step = int(self.waypoint.distance_between / self.cell_size)
        self.dynamic_objects = [
            Coin(
                map=map, position=waypoint, radius=radius,
                reward=reward + goal_r * (i == len(self.waypoints) - 1)
            ) for i, waypoint in enumerate(self.waypoints[1::step])
        ]

    def get_interactables(self):
        return [self.scene] + self.static_objects + self.dynamic_objects

    def create_waypoints(self):
        b = self.bounds

        path = self.global_planner.sample_path(self.trail_skeleton, debug=False)

        # b.y_max - 1 - iy, ix - b.x_min
        path[0] += b.x_min
        path[1] = b.y_max - 1 - path[1]

        x, y = self.get_xy(path[0], path[1])

        waypoints = np.array([x, y]).T

        half = max(int(self.n_agents_per_worker / 2), 1)
        step = int(len(waypoints) * 0.70 / half)

        initial_pose = np.zeros((6, self.n_agents_per_worker))
        for i in range(self.n_agents_per_worker):

            # w0 is the start point, w1 is used to determine the initial angle
            # use step*i + 5 instead of +1 because +1 is just 1px away
            w0 = waypoints[step*(i-half)                  ]
            w1 = waypoints[step*(i-half)+5*np.sign(i-half)]

            dx, dy = w1 - w0
            theta = np.arctan2(dy, dx) - np.pi / 2
            initial_pose[:, i] = [w0[0], w0[1], theta, 0, 0, 0]

        return waypoints, initial_pose

    def load_map_def(self, map_def):
        print "loading map ...",
        cwd = dirname(__file__)
        map_def_fn = "{}/../maps/{}.yaml".format(cwd, map_def)
        map_def = load_yaml(map_def_fn)

        # Load map structure
        fn = "{}/../maps/{}".format(cwd, map_def.map_structure_filename)
        map_def.map_structure = np.array(PIL.Image.open(fn))

        # Load MATLAB-pre-skeletonized trail map
        fn = "{}/../maps/{}".format(cwd, map_def.trail_skeleton_filename)
        map_def.trail_skeleton = np.array(PIL.Image.open(fn))

        for k, v in map_def.iteritems():
            if isinstance(v, dict):
                v = AttrDict(v)
            setattr(self, k, v)

        print "done."

    def save_rgb_map_in_png(self, fn):

        img = PIL.Image.fromarray(self.map_structure.astype(np.uint8))
        img = img.convert(mode='P')
        classes = self.class_id_to_rgb.keys()

        colors = np.zeros((len(classes), 3), np.uint8)
        for i in range(len(classes)):
            if i in classes:
                colors[i] = self.class_id_to_rgb[i]

        colors = colors.reshape(-1).tolist()

        img.putpalette(colors)

        img.save(fn)

    def colorize(self, labels):
        print "colorizing ...",
        img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        classes = np.unique(labels)
        for c in classes:
            img[labels == c] = self.class_id_to_rgb[c]
        print "done."
        return img

    def cvt_map_structure_to_rewards(self, labels):
        print "converting to reward ...",
        rewards = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.float32)
        classes = np.unique(labels)
        for c in classes:
            rewards[labels == c] = self.class_id_to_rewards[c]
        print "done."

        return rewards

    def get_class(self, x, y):
        b = self.bounds
        ix, iy = self.get_ixiy(x, y)
        c = self.map_structure[b.y_max - 1 - iy, ix - b.x_min]
        return c

    def in_tree(self, state):
        return self.get_class(*state[:2]) >= 5

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

        print "(cx, cy) = ({}, {})".format(cx, cy)
        print "(x_min, x_max) = ({}, {}), (y_min, y_max) = ({}, {})".format(
            self.bounds.x_min, self.bounds.x_max,
            self.bounds.y_min, self.bounds.y_max
        )
        print "(height, width) = ({}, {})".format(self.height, self.width)

    def contains(self, x, y):
        b = self.bounds
        ix, iy = self.get_ixiy(x, y)
        inside = (ix >= b.x_min) & (ix <= b.x_max - 1) \
            & (iy >= b.y_min) & (iy <= b.y_max - 1)
        return inside

    def get_xy(self, ix, iy):
        return (ix + 0.5) * self.cell_size, (iy + 0.5) * self.cell_size

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
        for obj in self.map.get_interactables():
            reward += obj.react(state)
        return reward
