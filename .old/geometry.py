import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


#  -------------------  MODULE NOT USED  -------------------  #


class Icosahedron:

    def __init__(self, g):

        self.granularity = g
        self.__vertices = None

    def vertices(self):

        r = (1 + np.sqrt(5)) / 2
        factors = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * 1e8  # compute as integers for better precision

        self.__vertices = np.concatenate((
            [[0, 1 * a, r * b] for a, b in factors],
            [[1 * a, r * b, 0] for a, b in factors],
            [[r * a, 0, 1 * b] for a, b in factors]
        ), axis=0)

        self._normalize_1e8()

        for _ in range(self.granularity):
            self._tessellate()

        self._true_normalize()

        to_delete = []
        for i, v in enumerate(self.__vertices):
            if v[2] < 0:
                to_delete.append(i)

        self.__vertices = np.delete(self.__vertices, to_delete, axis=0)

        return self.__vertices

    def _tessellate(self):

        adjacent = self._get_adjacent()
        omegas = self._get_omegas(adjacent)
        self.__vertices = np.concatenate((self.__vertices, omegas))
        self._normalize_1e8()
        self._remove_duplicates()

    def _remove_duplicates(self):

        self.__vertices = np.unique(self.__vertices, axis=0)

    def _get_adjacent(self):

        vertices = self.__vertices

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

    def _normalize_1e8(self):

        normalized = []
        for v in self.__vertices:
            mag = np.linalg.norm(v) / 1e8
            normalized.append(np.int64(np.divide(v, mag)))

        self.__vertices = np.array(normalized)

    def _true_normalize(self):

        normalized = []
        for v in self.__vertices:
            mag = np.linalg.norm(v)
            normalized.append(np.around(np.divide(v, mag), decimals=8))

        self.__vertices = np.array(normalized)

    @staticmethod
    def _get_omegas(adjacent):

        to_append = []
        for entry in adjacent:
            p1, p2 = entry["p1"], entry["p2"]
            ave_vector = np.int64(np.add(p1, p2) / 2)
            to_append.append(ave_vector)

        return to_append

    def plot(self):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        points = self.__vertices

        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        plt.show()
