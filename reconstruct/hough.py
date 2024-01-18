import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json

from .geometry import Icosahedron


class Hough:

    __slots__ = ("points", "__offset", "prime_range", "n_abs", "k_max", "dl", "n_hood", "max_rmse", "b", "xy", "A")

    def __init__(self, points: np.ndarray, n_abs, k_max, dl, n_hood=None):

        self.points, self.__offset = self.translate_to_origin(points)
        self.prime_range = self.get_prime_range(self.points)

        self.n_abs = n_abs
        self.k_max = k_max
        self.dl = dl
        self.n_hood = n_hood
        self.max_rmse = 0.1

        # define parameter discretization
        icosahedron = Icosahedron(g=5)
        self.b = icosahedron.vertices()
        self.xy = np.arange(*self.prime_range, dl)

        # initialize sparse accumulator
        self.A = {}

    def run_detection(self):

        lines = []
        used_points = []

        while True:

            points = list(self.get_accumulator_max())
            print(points)

            if self.n_abs and len(points) < self.n_abs:
                print("break")
                break

            line_params = self.svd_optimise(points)

            _exists = False
            for k in lines:
                if np.all([np.allclose(v, line_params[i]) for i, v in enumerate(k)]):
                    _exists = True

            if (line_params[2] < self.max_rmse) and not _exists:
                lines.append(line_params)

                for p in points:
                    used_points.append(p)

        return lines, used_points

    def find_lines(self):

        self.increment_accumulator()

        print("\033[96m[{}]\033[0m".format("Getting Lines".center(30)))

        self.sort_accumulator()
        lines, used_points = self.run_detection()

        self.plot_accumulator(lines)

    def plot_accumulator(self, lines):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        ax.set_zlim(-3, 3)

        v = np.arange(*self.prime_range, self.dl)

        for [point, b, _, _] in lines:
            ax.plot(point[0] + v * b[0], point[1] + v * b[1], point[2] + v * b[2])

        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])

        plt.show()

    def sort_accumulator(self):

        sorted_A = dict(sorted(self.A.items(), key=lambda x: len(x[1])))
        self.A = sorted_A

    def bin_accumulator(self):

        def _angle_diff(v, other_v):
            return np.arccos(np.dot(v, other_v))

        b_tolerance = 2 * np.pi / 180
        xy_tolerance = 0.2

        A = {}
        _ignore = []

        _iter = tqdm(self.A.items())
        for param, data in _iter:
            A[param] = data
            other_params = {o_p: o_d for o_p, o_d in self.A.items() if o_p not in A.keys() and o_p not in _ignore}

            #angles = _angle_diff(np.array(param[2]), np.array([np.array(v) for v in other_params.values()]))

            for op, od in other_params.items():
                b1 = np.array(param[2])
                b2 = np.array(op[2])

                param_a = list(param)
                op_a = list(op)

                sep_angle = np.arccos(np.dot(b1, b2))
                dx = np.abs(op_a[0] - param_a[0])
                dy = np.abs(op_a[1] - param_a[1])

                if (sep_angle < b_tolerance) and (dx < xy_tolerance) and (dy < xy_tolerance):
                    _ignore.append(op)
                    A[param].update(od)

        self.A = A

    def get_accumulator_max(self):

        params, points = list(self.A.items())[-1]

        try:
            del self.A[params]
        except KeyError:
            pass

        return points

    def increment_accumulator(self, points=None):

        if points is None:
            points = self.points

        self.A = {}

        run_points = []
        _iter = tqdm(points)
        for i, p in enumerate(_iter):
            run_points.append(i)
            other_points = {k: a for k, a in enumerate(points) if k not in run_points}

            for j, (opi, op) in enumerate(other_points.items()):

                b = tuple(self.get_b(p, op))
                xp = self.x_prime(p, b)
                yp = self.y_prime(p, b)

                _params = (xp, yp, b)
                try:
                    self.A[_params].add(i)
                except KeyError:
                    self.A[_params] = {i, opi}

            _iter.set_description("Incrementing Accumulator ({:.1f} MB)".format(sys.getsizeof(self.A) / 1e6))

        self.bin_accumulator()

        with open("A.json", "w") as f:
            A_stringify = {str(k): list(v) for k, v in self.A.items()}
            json.dump(A_stringify, f)

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
        return xp

    def y_prime(self, p, b):

        yp = - b[0] * b[1] / (1 + b[2]) * p[0] + (1 - (b[1] ** 2 / (1 + b[2]))) * p[1] - (b[1] * p[2])
        return yp

    def get_b(self, p1, p2):

        b = p2 - p1 if (p2[2] > p1[2]) else p1 - p2
        b_norm = b / np.linalg.norm(b)

        return b_norm

    def svd_optimise(self, indices):

        points = self.points[indices]
        mean = points.mean(axis=0)
        _, _, vv = np.linalg.svd(points - mean, full_matrices=False)

        rmse = self.get_rmse(points, mean, vv[0])

        return [np.around(mean, 4), np.around(vv[0], 4), np.around(rmse, 4), np.int64(len(points))]

    def get_rmse(self, points, mean, direction):

        errors = 0
        for p in points:
            expect_z = p[2]
            expect_y = p[1]
            expect_x = p[0]

            if direction[2] != 0:
                calc_point = mean + (expect_z / direction[2]) * direction
            elif direction[1] != 0:
                calc_point = mean + (expect_y / direction[1]) * direction
            elif direction[0] != 0:
                calc_point = mean + (expect_x / direction[0]) * direction
            else:
                raise ValueError("Line has invalid direction vector b = (0, 0, 0).")

            dx = calc_point[0] - p[0]
            dy = calc_point[1] - p[1]

            errors += dx ** 2 + dy ** 2

        n = len(points)
        rmse = np.sqrt(errors / n if n != 0 else 1)
        return rmse
