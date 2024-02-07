import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from tqdm import tqdm
import sys
import time


class Transform:

    def __init__(self, points: np.ndarray, n_abs, k_max, btol, xytol, min_angle, plot=False):

        self.points = points
        self.prime_range = self.get_prime_range(self.points)

        self.n_abs = n_abs
        self.k_max = k_max
        self.dl = 1
        self.max_rmse = 0.1

        self.btol = btol
        self.xytol = xytol
        self.min_angle = min_angle

        self.plot = plot

        # initialize sparse accumulator
        self.A = {}

    def run_detection(self):

        lines = []
        used_points = []

        while True:

            try:
                points = list(self.get_accumulator_max())
            except TypeError:
                break

            if self.n_abs and len(points) < self.n_abs:
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

        time_now = time.time()

        self.increment_accumulator()

        self.sort_accumulator()
        lines, used_points = self.run_detection()

        print("\n--- Complete in {:.4f} seconds ---\n".format(time.time() - time_now))

        if self.plot is True:
            self.plot_accumulator(lines)
        return lines

    def plot_accumulator(self, lines):

        fig = plt.figure(figsize=(8, 8))
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

        def _dist_from_rad(rad):
            return np.sqrt(2 * (1 - np.cos(rad)))

        A = {}
        _ignore = set()

        query_params = {op: od for op, od in self.A.items()}

        xy_kdtree = KDTree([[p[0], p[1]] for p in query_params])
        b_kdtree = KDTree([list(p[2]) for p in query_params])

        _iter = tqdm(self.A.items())
        _iter.set_description("Binning".ljust(35))
        for i, (param, data) in enumerate(_iter):

            if i in _ignore:
                continue

            _ignore.add(i)

            close_xy_indices = set(xy_kdtree.query_radius([param[0:2]], r=self.xytol)[0].tolist())
            close_b_indices = set(b_kdtree.query_radius([param[2]], r=_dist_from_rad(self.btol))[0].tolist())

            indices = close_xy_indices.intersection(close_b_indices)
            indices.discard(_ignore)

            if indices:
                A[param] = data

                _ignore.update(indices)
                close = [list(query_params)[j] for j in indices]

                for key in close:
                    A[param].update(query_params[key])

        self.A = A

    def get_accumulator_max(self):

        try:
            params, points = list(self.A.items())[-1]
        except IndexError:
            return None

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

                if np.isclose(op[2], p[2]):
                    continue

                b = tuple(self.get_b(p, op))
                xp = self.get_xprime(p, b)
                yp = self.get_yprime(p, b)

                if b[2] <= 0.01:
                    continue

                _params = (xp, yp, b)
                try:
                    self.A[_params].add(i)
                except KeyError:
                    self.A[_params] = {i, opi}

            _iter.set_description("Incrementing Accumulator ({:.1f} KB)".format(sys.getsizeof(self.A) / 1e3))

        self.bin_accumulator()

    def svd_optimise(self, indices):

        points = self.points[indices]
        mean = points.mean(axis=0)
        _, _, vv = np.linalg.svd(points - mean, full_matrices=False)

        rmse = self.get_rmse(points, mean, vv[0])

        return [np.around(mean, 4), np.around(vv[0], 4), np.around(rmse, 4), np.int64(len(points))]

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

    @staticmethod
    def get_rmse(points, mean, direction):

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

    @staticmethod
    def get_xprime(p, b):

        xp = (1 - b[0] ** 2 / (1 + b[2])) * p[0] - b[0] * b[1] / (1 + b[2]) * p[1] - b[0] * p[2]
        return xp

    @staticmethod
    def get_yprime(p, b):

        yp = - b[0] * b[1] / (1 + b[2]) * p[0] + (1 - (b[1] ** 2 / (1 + b[2]))) * p[1] - (b[1] * p[2])
        return yp

    @staticmethod
    def get_b(p1, p2):

        b = p2 - p1 if (p2[2] > p1[2]) else p1 - p2
        b_norm = b / np.linalg.norm(b)

        return b_norm
