import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


class Hough:

    def __init__(self, points, n_min, k_max, dx):

        self.points = points
        self.n_min = n_min
        self.k_max = k_max
        self.dx = dx
        self._vertices = None

        # define parameter discretization
        self.b_params = self.b_get_discretization(5)

    def b_get_discretization(self, n):

        r = (1 + np.sqrt(5)) / 2
        factors = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * 1e8

        self._vertices = np.concatenate((
            [[0, 1 * a, r * b] for a, b in factors],
            [[1 * a, r * b, 0] for a, b in factors],
            [[r * a, 0, 1 * b] for a, b in factors]
        ))

        self.normalize_1e8()

        for _ in range(n):
            self.tessellate()

        self.true_normalize()

        self.plot(self._vertices)

        return self._vertices

    def tessellate(self):

        adjacent = self.get_adjacent()
        omegas = self.get_omegas(adjacent)
        self._vertices = np.concatenate((self._vertices, omegas))
        self.normalize_1e8()
        self.remove_duplicates()

    def get_omegas(self, adjacent):

        to_append = []
        for entry in adjacent:
            p1, p2 = entry["p1"], entry["p2"]
            ave_vector = np.int64(np.add(p1, p2) / 2)
            to_append.append(ave_vector)

        return to_append

    def remove_duplicates(self):

        self._vertices = np.unique(self._vertices, axis=0)

    def get_adjacent(self):

        vertices = self._vertices

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
        for v in self._vertices:
            mag = np.linalg.norm(v) / 1e8
            normalized.append(np.int64(np.divide(v, mag)))

        self._vertices = np.array(normalized)

    def true_normalize(self):

        normalized = []
        for v in self._vertices:
            mag = np.linalg.norm(v)
            normalized.append(np.around(np.divide(v, mag), decimals=8))

        self._vertices = np.array(normalized)

    @staticmethod
    def sph_to_cart(theta, phi):

        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)

        return [x, y, z]

    def get_prime_plane(self):
        pass

    @staticmethod
    def plot(points):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        plt.show()

