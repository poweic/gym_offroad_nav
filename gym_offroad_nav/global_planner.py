import cv2
import PIL.Image
import numpy as np
from attrdict import AttrDict

def breadth_first_search(graph, begin, end, debug=False):
    assert graph.ndim == 2
    visited = np.zeros_like(graph)
    H, W = graph.shape
    
    nodes = [AttrDict(x=begin[0], y=begin[1], p=None)]
    visited[begin[1], begin[0]] = True

    # debugging
    if debug:
        color_g = cv2.cvtColor(graph.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
        color_g[begin[1], begin[0], :] = [165, 0, 255]
        color_g[end[1], end[0], :] = [165, 0, 255]
    
    reacheable = False
    N = 1

    i = 0
    while i < len(nodes):
        node = nodes[i]
        i += 1

        # alias for node.x and node.y
        x, y = node.x, node.y

        if (x, y) == end:
            reacheable = True
            break

        for xx in range(x-N, x+N+1):
            for yy in range(y-N, y+N+1):
                if not (0 <= yy < H and 0 <= xx < W):
                    continue

                if graph[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    nodes.append(AttrDict(
                        x = xx,
                        y = yy,
                        p = node
                    ))

    if not reacheable:
        return None

    path = []

    while node is not None:

        if debug:
            color_g[node.y, node.x, :] = [0, 0, 255]
            cv2.imshow("graph", cv2.resize(color_g, (400, 400), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(1)

        path.append((node.x, node.y))
        node = node.p

    if debug:
        cv2.waitKey(100)

    return path

class GlobalPlanner(object):
    def __init__(self):
        self.rng = np.random.RandomState()

    def seed(self, rng):
        self.rng = rng

    def sample_path(self, graph, max_trial=40, debug=False):
        path = None

        # if BFS cannot find path between begin and end, a None value is
        # returned. Keep resample (begin, end) until we can found a path
        while path is None:
            begin, end = self.sample_points(graph, max_trial)
            path = breadth_first_search(graph > 0, begin, end, debug)

        return np.array(path).T

    def sample_points(self, img, max_trial=20):
        assert img.ndim == 2

        img = img.astype(np.float32)

        H, W = img.shape
        dist_x = np.sum(img, axis=0)
        dist_x = dist_x / np.sum(dist_x)

        trial = 0

        def sample_one_point():
            x = self.rng.choice(range(W), p=dist_x)
            dist_y = img[:, x] / np.sum(img[:, x])
            y = self.rng.choice(range(H), p=dist_y)
            assert img[y, x] > 0
            return x, y

        begin = sample_one_point()
        max_dist = 0

        for i in range(max_trial):
            x, y = sample_one_point()

            dist = np.linalg.norm(np.array([x - begin[0], y - begin[1]]))

            if dist > max_dist:
                end = (x, y)
                max_dist = dist

        return begin, end
