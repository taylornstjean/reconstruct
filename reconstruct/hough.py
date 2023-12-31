import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left
from tqdm import tqdm
import sys

from .geometry import Icosahedron


class Hough:

    __slots__ = ("points", "__offset", "prime_range", "n_abs", "k_max", "dl", "n_hood", "b", "xy", "A")

    def __init__(self, points: np.ndarray, n_abs, k_max, dl, n_hood):

        self.points, self.__offset = self.translate_to_origin(points)
        self.prime_range = self.get_prime_range(self.points)

        self.n_abs = n_abs
        self.k_max = k_max
        self.dl = dl
        self.n_hood = n_hood

        # define parameter discretization
        icosahedron = Icosahedron(g=4)
        self.b = icosahedron.vertices()
        self.xy = np.arange(*self.prime_range, dl)

        # initialize sparse accumulator
        self.A = {}

    def plot_accumulator(self):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        ax.set_zlim(-3, 3)

        v = np.arange(*self.prime_range, self.dl)

        lines = []

        self.sort_accumulator()

        print("\033[96m[{}]\033[0m".format("Getting Lines".center(30)))

        for _ in range(self.k_max):
            vote, points = self.get_accumulator_max()

            if self.n_abs and vote < self.n_abs:
                break

            lines.append(self.svd_optimise(points))

        for [point, b] in lines:
            ax.plot(point[0] + v * b[0], point[1] + v * b[1], point[2] + v * b[2])

        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])

        plt.show()

    def sort_accumulator(self):

        print("\033[96m[{}]\033[0m".format("Running Accumulator Sort".center(30)))

        sorted_A = dict(sorted(self.A.items(), key=lambda x: x[1][0]))
        self.A = sorted_A

    def get_accumulator_max(self):

        maximum, value = list(self.A.items())[-1]
        vote = value[0]
        points = value[1]

        self.clear_nhood(maximum)

        return vote, points

    def clear_nhood(self, params):

        xp = params[0]
        yp = params[1]
        b = self.b_get_neighbors(self.b[params[2]])

        for i in np.arange(xp - self.n_hood / 2, xp + self.n_hood / 2, 1):
            for j in np.arange(yp - self.n_hood / 2, yp + self.n_hood / 2, 1):
                for k in b:
                    try:
                        del self.A[(int(i), int(j), int(k))]
                    except KeyError:
                        pass

    def increment_accumulator(self):

        _iter = tqdm(self.points)
        for i, p in enumerate(_iter):
            for j, b in enumerate(self.b):

                xp_i = self.x_prime(p, b)
                yp_i = self.y_prime(p, b)

                try:
                    self.A[(xp_i, yp_i, j)][0] += 1
                    self.A[(xp_i, yp_i, j)][1].append(i)
                except KeyError:
                    self.A[(xp_i, yp_i, j)] = [1, [i]]

            _iter.set_description("Incrementing Accumulator ({:.1f} MB)".format(sys.getsizeof(self.A) / 1e6))

    @staticmethod
    def point_from_prime(xp, yp, b):

        x_coeff = np.array([
            1 - (b[0] ** 2) / (1 + b[2]),
            -b[0] * b[1] / (1 + b[2]),
            -b[0]
        ])
        y_coeff = np.array([
            -b[0] * b[1] / (1 + b[2]),
            1 - (b[1] ** 2) / (1 + b[2]),
            -b[1]
        ])

        point = np.add(xp * x_coeff, yp * y_coeff)

        return point

    @staticmethod
    def translate_to_origin(points):

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        c = np.array([
            np.max(xs) + np.min(xs),
            np.max(ys) + np.min(ys),
            np.max(zs) + np.min(zs)
        ]) / 2

        return np.subtract(points, c), c

    @staticmethod
    def get_prime_range(points):

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        d = np.array([
            np.max(xs) - np.min(xs),
            np.max(ys) - np.min(ys),
            np.max(zs) - np.min(zs)
        ]) / 2

        half_mag = np.linalg.norm(d / 2)

        return [-half_mag * 2, half_mag * 2]  # add buffer

    def x_prime(self, p, b):
        xp = (1 - b[0] ** 2 / (1 + b[2])) * p[0] - b[0] * b[1] / (1 + b[2]) * p[1] - b[0] * p[2]
        index = self.xy_discretize(xp)
        return index

    def y_prime(self, p, b):
        yp = - b[0] * b[1] / (1 + b[2]) * p[0] + (1 - (b[1] ** 2 / (1 + b[2]))) * p[1] - (b[1] * p[2])
        index = self.xy_discretize(yp)
        return index

    def xy_discretize(self, v):

        pos = bisect_left(self.xy.tolist(), v)

        if pos == 0:
            return 0
        if pos == np.shape(self.xy)[0]:
            return np.shape(self.xy)[0] - 1

        before = self.xy[pos - 1]
        after = self.xy[pos]

        if after - v < v - before:
            return pos
        return pos - 1

    def b_get_neighbors(self, v):

        angles = {}
        for i, d in enumerate(self.b):
            dp = d[0] * v[0] + d[1] * v[1] + d[2] * v[2]
            angles[i] = dp

        s_angles = {k: v for k, v in sorted(angles.items(), key=lambda item: item[1])}

        pos = list(s_angles.keys())[-self.n_hood:]

        return pos

    def svd_optimise(self, indices):

        points = self.points[indices]

        mean = points.mean(axis=0)
        _, _, vv = np.linalg.svd(points - mean, full_matrices=False)

        return [mean, vv[0]]

