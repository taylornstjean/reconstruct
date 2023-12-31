import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from bisect import bisect_left


class Hough:

    def __init__(self, points: np.ndarray, n_min, k_max, dx, n_hood):

        self.points, self.__offset = self.translate_to_origin(points)
        self.prime_range = self.get_prime_range(self.points)

        self.n_min = n_min
        self.k_max = k_max
        self.dx = dx
        self.n_hood = n_hood

        # initialize vertices attr
        self.vertices = None

        # define parameter discretization
        self.b_discrete = self.b_get_discretization(5)
        self.xp_discrete = self.yp_discrete = np.arange(*self.prime_range, dx)

        # initialize sparse accumulator
        self.A = {}

    def plot_accumulator(self):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        ax.set_zlim(-3, 3)

        v = np.arange(*self.prime_range, self.dx)

        lines = []

        for _ in range(self.k_max):
            params, val = self.get_accumulator_max()

            if val < self.n_min:
                break

            xp = self.xp_discrete[params[0]]
            yp = self.xp_discrete[params[1]]
            b = self.b_discrete[params[2]]

            point = self.point_from_prime(xp, yp, b)

            # print(f"Plotting line --> [{point[0]}, {point[1]}, {point[2]}] + t[{b[0]}, {b[1]}, {b[2]}]")

            lines.append([point, b])

        for [point, b] in lines:
            ax.plot(point[0] + v * b[0], point[1] + v * b[1], point[2] + v * b[2])

        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])

        plt.show()

    def get_accumulator_max(self):

        maximum = max(self.A, key=self.A.get)
        value = self.A[maximum]

        self.clear_nhood(maximum)

        return maximum, value

    def clear_nhood(self, params):

        xp = params[0]
        yp = params[1]
        b = self.b_get_neighbors(self.b_discrete[params[2]])

        for i in np.arange(xp - self.n_hood / 2, xp + self.n_hood / 2, 1):
            for j in np.arange(yp - self.n_hood / 2, yp + self.n_hood / 2, 1):
                for k in b:
                    try:
                        del self.A[(int(i), int(j), int(k))]
                    except KeyError:
                        pass

    def increment_accumulator(self):

        for p in self.points:
            # print("\033[92m", "Point --> ", p, "\033[0m")
            # print("-------------------------------------------------------------------")
            for i, b in enumerate(self.b_discrete):
                # print("Index, Direction --> ", i, ",", b)

                xp_i = self.x_prime(p, b)
                yp_i = self.y_prime(p, b)

                # print(f'\033[94m(x\', y\', i), ({xp_i}, {yp_i}, {i}), ({self.xp_discrete[xp_i]}, {self.yp_discrete[yp_i]}, {b})\033[0m')

                try:
                    self.A[(xp_i, yp_i, i)] += 1
                    # print(f"Incremented ({xp_i}, {yp_i}, {i}) to {self.A[(xp_i, yp_i, i)]}")
                except KeyError:
                    self.A[(xp_i, yp_i, i)] = 1
                    # print(f"\033[93mNew Entry ({xp_i}, {yp_i}, {i})\033[0m")

                # print("-------------------------------------------------------------------")

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

    def b_get_discretization(self, n):

        r = (1 + np.sqrt(5)) / 2
        factors = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * 1e8  # compute as integers for better precision

        self.vertices = np.concatenate((
            [[0, 1 * a, r * b] for a, b in factors],
            [[1 * a, r * b, 0] for a, b in factors],
            [[r * a, 0, 1 * b] for a, b in factors]
        ), axis=0)

        self.normalize_1e8()

        for _ in range(n):
            self.tessellate()

        self.true_normalize()

        to_delete = []
        for i, v in enumerate(self.vertices):
            if v[2] < 0:
                to_delete.append(i)

        self.vertices = np.delete(self.vertices, to_delete, axis=0)

        return self.vertices

    def tessellate(self):

        adjacent = self.get_adjacent()
        omegas = self.get_omegas(adjacent)
        self.vertices = np.concatenate((self.vertices, omegas))
        self.normalize_1e8()
        self.remove_duplicates()

    @staticmethod
    def get_omegas(adjacent):

        to_append = []
        for entry in adjacent:
            p1, p2 = entry["p1"], entry["p2"]
            ave_vector = np.int64(np.add(p1, p2) / 2)
            to_append.append(ave_vector)

        return to_append

    def remove_duplicates(self):

        self.vertices = np.unique(self.vertices, axis=0)

    def get_adjacent(self):

        vertices = self.vertices

        adjacent_v = []
        kdtree = KDTree(vertices)
        for i, point in enumerate(vertices):
            dist, points = kdtree.query(np.array([point]), 7)
            for d, p in zip(dist[0][1::], vertices[points[0][1::]]):
                adjacent_v.append({"dist": np.int64(d), "p1": np.int64(point), "p2": np.int64(p)})

        average_dist = np.average([a["dist"] for a in adjacent_v])
        to_delete = []
        for i, entry in enumerate(adjacent_v):
            if entry["dist"] > (1.1 * average_dist) or entry["dist"] < (0.8 * average_dist):
                to_delete.append(i)

        for i in sorted(to_delete, reverse=True):
            del adjacent_v[i]

        return adjacent_v

    def normalize_1e8(self):

        normalized = []
        for v in self.vertices:
            mag = np.linalg.norm(v) / 1e8
            normalized.append(np.int64(np.divide(v, mag)))

        self.vertices = np.array(normalized)

    def true_normalize(self):

        normalized = []
        for v in self.vertices:
            mag = np.linalg.norm(v)
            normalized.append(np.around(np.divide(v, mag), decimals=8))

        self.vertices = np.array(normalized)

    @staticmethod
    def sph_to_cart(theta, phi):

        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)

        return [x, y, z]

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

        # print(f"d = {d}")

        half_mag = np.linalg.norm(d / 2)

        return [-half_mag * 2, half_mag * 2]  # add 10% buffer

    def x_prime(self, p, b):
        # print(f"\nx-Prime Received Input --> p = ({p[0]}, {p[1]}, {p[2]}), b = ({b[0]}, {b[1]}, {b[2]})")
        xp = (1 - b[0] ** 2 / (1 + b[2])) * p[0] - b[0] * b[1] / (1 + b[2]) * p[1] - b[0] * p[2]
        index = self.xy_discretize(xp)
        # print(f"Output --> i = {index}, x\' = {xp}")
        return index

    def y_prime(self, p, b):
        # print(f"y-Prime Received Input --> p = ({p[0]}, {p[1]}, {p[2]}), b = ({b[0]}, {b[1]}, {b[2]})")
        yp = - b[0] * b[1] / (1 + b[2]) * p[0] + (1 - (b[1] ** 2 / (1 + b[2]))) * p[1] - (b[1] * p[2])
        index = self.xy_discretize(yp)
        # print(f"Output --> i = {index}, y\' = {yp}\n")
        return index

    def xy_discretize(self, v):

        pos = bisect_left(self.xp_discrete.tolist(), v)

        if pos == 0:
            return 0
        if pos == np.shape(self.xp_discrete)[0]:
            return np.shape(self.xp_discrete)[0] - 1

        before = self.xp_discrete[pos - 1]
        after = self.xp_discrete[pos]

        if after - v < v - before:
            return pos
        return pos - 1

    def b_get_neighbors(self, v):

        angles = {}
        for i, d in enumerate(self.b_discrete):
            dp = d[0] * v[0] + d[1] * v[1] + d[2] * v[2]
            angles[i] = dp

        s_angles = {k: v for k, v in sorted(angles.items(), key=lambda item: item[1])}

        pos = list(s_angles.keys())[-self.n_hood:]

        return pos

    @staticmethod
    def plot(points):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        plt.show()

