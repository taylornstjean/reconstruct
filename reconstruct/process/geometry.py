import numpy as np
from itertools import combinations


class HyperPlane:

    """Not used for now."""

    def __init__(self, normal: np.ndarray):

        self._n = normal
        self._a, self._b, self._c = self._get_orthogonals()

    def _get_orthogonals(self) -> np.ndarray:

        w = {
            1: np.add(self._n, np.array([1, 0, 0, 0])),
            2: np.add(self._n, np.array([0, 1, 0, 0])),
            3: np.add(self._n, np.array([0, 0, 1, 0]))
        }

        u = np.array([
            np.subtract(w[i], (np.dot(self._n.T, w[i]) / (np.linalg.norm(self._n) ** 2)) * self._n) for i in [1, 2, 3]
        ])

        return u


class FitHyperPlane:

    def __init__(self, points):

        self._points = points

    def params(self) -> tuple:

        b_list = []
        for p1, p2 in combinations(self._points, 2):
            b_list.append((p2 - p1) / np.linalg.norm((p2 - p1)))

        b = np.average(np.array(b_list), axis=0)
        p = np.average(self._points, axis=0)

        b_norm = np.array(b / np.linalg.norm(b))
        p_norm = np.array(p - (b_norm / b_norm[2]) * p[2])

        return p_norm, b_norm

    def rmse(self):

        p, b = self.params()

        dists = []
        for point in self._points:
            pd = point - p
            proj = np.dot(pd, b)
            dists.append(np.linalg.norm(pd - proj * b))

        rmse = np.sqrt(sum([d ** 2 for d in dists]) / len(dists))

        return rmse
