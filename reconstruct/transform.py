import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
import sys
import time
from .renderer import Plot3D


class Transform:
    """Houses the line detector. Initialize with a 3D point cloud and run ``find_lines`` method to execute the algorithm."""

    def __init__(self, points: np.ndarray, n_abs, k_max, plate_spacing, pos_resolution, min_angle, max_rmse, plot=False) -> None:

        """
        :param points: 3D input points to transform.
        :type points: np.ndarray

        :param n_abs: The minimum number of points that constitute a path detection.
        :type n_abs: int

        :param k_max: The maximum number of lines to detect.
        :type k_max: int

        :param plate_spacing: The vertical spacing in meters between each tracker in the main detector.
        :type plate_spacing: int | float

        :param pos_resolution: The positional resolution in meters of the trackers in the detector.
        :type pos_resolution: int | float

        :param min_angle: The minimum angle above the `xy` plane of the direction vector `b` that defines a line. Any `b` with an angle above the `xy` plane greater than ``min_angle`` will have their associated line discarded.
        :type min_angle: int | float

        :param max_rmse: The maximum RMSE value at which a path detection is valid.
        :type max_rmse: int | float

        :param plot: Plots the resulting lines if ``True``, default is ``False``.
        :type plot: bool
        """

        self.points = points

        # plot point spacing
        self.dl = 1

        # find total range of x' and y' values
        self.prime_range = self._get_prime_range(self.points)

        self.plate_spacing = plate_spacing
        self.pos_resolution = pos_resolution
        self.n_abs = n_abs
        self.k_max = k_max
        self.btol = self._get_btol()
        self.xytol = self._get_xytol()
        self.min_angle = min_angle
        self.max_rmse = max_rmse
        self.plot = plot

        # initialize sparse accumulator
        self.A = {}

    def _get_btol(self):

        """Calculate the b-tolerance value based on the geometry of the detector."""

        max_d = np.sqrt(2 * self.pos_resolution ** 2)
        btol = np.arctan(max_d / self.plate_spacing) / 4  # add buffer zone

        return btol

    def _get_xytol(self):

        """Calculate the xy-tolerance value based on the geometry of the detector."""

        xytol = self.pos_resolution * 30  # add buffer zone

        return xytol

    def _run_detection(self) -> tuple[list[list], list]:

        """Translates the accumulator into a list of best fit lines."""

        lines = []
        used_points = []

        while True:

            if len(lines) >= self.k_max:
                # break if the number of desired lines have been found
                break

            try:
                # get the maximum voted line in the accumulator
                points = list(self._get_accumulator_max())
            except TypeError:
                # break if the accumulator has been drained
                break

            if self.n_abs and len(points) < self.n_abs:
                # break if the number of points is not sufficient to constitute a detection
                break

            print(points)

            # get line params using SVD
            line_params = self._svd_optimise(points)

            # check if the line is a duplicate
            _exists = False
            for k in lines:
                if np.all([np.allclose(v, line_params[i]) for i, v in enumerate(k)]):
                    _exists = True

            # append line to database if it is not a duplicate and has low enough RMSE
            print(line_params[2])
            if (line_params[2] < self.max_rmse) and not _exists:
                lines.append(line_params)

                # record used points
                for p in points:
                    used_points.append(p)

        return lines, used_points

    def find_lines(self) -> list:

        """Run the line-finding algorithm on stored data. This is the primary entry point to the algorithm."""

        time_now = time.time()

        # run the populator
        self._populate_accumulator()

        # run the sorter
        self._sort_accumulator()
        lines, used_points = self._run_detection()

        print("\n--- Complete in {:.4f} seconds ---\n".format(time.time() - time_now))

        if self.plot is True:
            plotter = Plot3D()
            plotter.lines([line[0:2] for line in lines], self.prime_range, self.dl)
            plotter.points(self.points)
            plotter.save("plot_transform.html")
            plotter.show()

        return lines

    def _sort_accumulator(self):

        """Sort the accumulator in ascending order based on the vote count."""

        sorted_A = dict(sorted(self.A.items(), key=lambda x: len(x[1])))
        self.A = sorted_A

    def _bin_accumulator(self) -> None:

        """Bin the accumulator. Converts the list of line segments generated by the populator into a set of line candidates by generating composite lines from segments with similar parameters."""

        def _dist_from_rad(rad):
            """Convert radians to the radius of a circle the angle would draw out."""
            return np.sqrt(2 * (1 - np.cos(rad)))

        def _span_all_layers(array):
            """Check if there is at least one detected point on every layer for a line."""
            expected = set(np.arange(-2, 3, 1) * self.plate_spacing)
            seen = set(array)
            if seen.union(expected) != expected:
                return False
            return True

        A = {}
        _ignore = set()

        # define a KD tree for both b and xy for fast distance queries
        query_params = {op: od for op, od in self.A.items()}

        xy_kdtree = KDTree([[p[0], p[1]] for p in query_params])
        b_kdtree = KDTree([list(p[2]) for p in query_params])

        _iter = tqdm(self.A.items())
        _iter.set_description("Binning".ljust(35))

        # iterate over each entry in the accumulator
        for i, (param, data) in enumerate(_iter):

            if i in _ignore:
                # ignore any previously used line segments
                continue

            _ignore.add(i)

            # find all other line segments that lie within the position and angle tolerances
            close_xy_indices = set(xy_kdtree.query_radius([param[0:2]], r=self.xytol)[0].tolist())
            close_b_indices = set(b_kdtree.query_radius([param[2]], r=_dist_from_rad(self.btol))[0].tolist())

            # isolate line segments that are within both xy and b tolerance
            indices = close_xy_indices.intersection(close_b_indices)

            # ignore any previously used line segments
            indices.difference_update(_ignore)

            # generate composite lines
            if indices:

                # convert indices to line params
                close = [list(query_params)[j] for j in indices]

                # check if line has at least one detection point on each layer
                p_indices = list(set().union(*[query_params[key] for key in close]))
                # line_zs = [np.around(self.points[k][2], 2) for k in p_indices]

                # if not _span_all_layers(line_zs):
                #     continue

                # update new binned accumulator
                A[param] = data
                _ignore.update(indices)

                for key in close:
                    A[param].update(query_params[key])

        # replace the original accumulator with the new binned version
        self.A = A

    def _get_accumulator_max(self) -> tuple | None:

        """Get the maximum value of the accumulator. Must run ``sort_accumulator`` once for this function to work."""

        try:
            # get last value of accumulator (assumes sorted)
            params, points = list(self.A.items())[-1]
        except IndexError:
            # the accumulator is empty
            return None

        # delete the entry in the accumulator
        try:
            del self.A[params]
        except KeyError:
            pass

        return points

    def _populate_accumulator(self, points=None):

        """
        Increment the voting array using an input 3D point cloud.

        :param points: The input point cloud to use when populating the accumulator. Defaults to ``self.points``.
        :type points: np.ndarray
        """

        if points is None:
            points = self.points

        # reset the accumulator
        self.A = {}

        run_points = []
        _iter = tqdm(points)

        c = 299792458  # m/s
        max_time_delta = np.sqrt((100 / c) ** 2 + (0.8 / c) ** 2) * 1e9 / 2 * 1000  # nanoseconds

        # iterate over each point
        for i, p in enumerate(_iter):
            run_points.append(i)
            other_points = {k: a for k, a in enumerate(points) if k not in run_points}

            # iterate over every other point (ignoring previously run points)
            for j, (opi, op) in enumerate(other_points.items()):

                if np.isclose(op[2], p[2]):
                    # ignore if the points are the same
                    continue

                if np.abs(op[3] - p[3]) > max_time_delta:
                    # ignore if time difference is too great
                    continue

                # generate line segment between points p and op and convert to the primed coordinate frame
                b = tuple(self._get_b(p[0:3], op[0:3]))
                xp = self._get_xprime(p[0:3], b[0:3])
                yp = self._get_yprime(p[0:3], b[0:3])

                if b[2] <= 0.01:
                    continue

                # append parameters to the accumulator
                _params = (xp, yp, b)
                try:
                    self.A[_params].add(i)
                except KeyError:
                    self.A[_params] = {i, opi}

            # display the current memory usage
            _iter.set_description("Incrementing Accumulator ({:.1f} KB)".format(sys.getsizeof(self.A) / 1e3))

        # run the binner
        self._bin_accumulator()

    def _svd_optimise(self, indices) -> list:

        """
        Compute an optimized best fit line using SVD from a given set of points.

        :param indices: The indices of the points in the local point list.
        :type indices: list[int]
        """

        # convert indices to points
        points = self.points[indices]

        # calculate the mean and run SVD
        mean = points.mean(axis=0)
        _, _, vv = np.linalg.svd(points - mean, full_matrices=False)

        # get the RMSE for the calculated line
        rmse = self._get_rmse(points, mean, vv[0])

        return [np.around(mean, 4), np.around(vv[0], 4), np.around(rmse, 4), np.int64(len(points))]

    @staticmethod
    def _point_from_prime(xp, yp, b) -> np.ndarray:

        """
        Compute the [`x`, `y`, `z`] coordinate from the primed coordinates [`x'`, `y'`] and direction vector `b`.

        :param xp: The input `x'` point.
        :type xp: int | float

        :param yp: The input `y'` point.
        :type yp: int | float

        :param b: The input direction vector in 3D.
        :type b: list | tuple | np.ndarray
        """

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
    def _get_prime_range(points):

        """
        Calculate the domain for the prime coordinates.

        :param points: The input points in 3D.
        :type points: np.ndarray
        """

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        d = np.array([
            np.max(xs) - np.min(xs),
            np.max(ys) - np.min(ys),
            np.max(zs) - np.min(zs)
        ]) / 2

        half_mag = np.linalg.norm(d / 2)

        return [-half_mag * 2.2, half_mag * 2.2]

    @staticmethod
    def _get_rmse(points, mean, b):

        """
        Compute the root mean standard error (RMSE) value given a line and the set of points on the line that led to its detection.

        :param points: The input points in 3D.
        :type points: np.ndarray

        :param mean: The mean [`x`, `y`, `z`] value of the line.
        :type mean: np.ndarray

        :param b: The input direction vector in 3D.
        :type b: list | tuple | np.ndarray
        """

        errors = 0  # running sum of error
        for p in points:

            # ignore time diff for RMSE calculation
            # consider implementing Gram-Schmidt Orthogonalization to include timings
            distance = np.abs(np.linalg.norm(np.cross(p[0:3] - mean[0:3], b[0:3])))

            errors += distance ** 2

        n = len(points)
        rmse = np.sqrt(errors / n if n != 0 else 1)  # normalize rmse
        return rmse

    @staticmethod
    def _get_xprime(p, b):

        """
        Compute the `x'` value for a given point ``p`` and direction ``b`` pair. The value `x'` is calculated using the function:

        .. math::

            x'=\\left(1-\\frac{b^2_x}{1+b_z}\\right)p_x-\\left(\\frac{b_xb_y}{1+b_z}\\right)p_y-b_xp_z

        :param p: The input point in 3D.
        :type p: list | tuple | np.ndarray

        :param b: The input direction vector in 3D.
        :type b: list | tuple | np.ndarray
        """

        xp = (1 - b[0] ** 2 / (1 + b[2])) * p[0] - b[0] * b[1] / (1 + b[2]) * p[1] - b[0] * p[2]
        return xp

    @staticmethod
    def _get_yprime(p, b):

        """
        Compute the `y'` value for a given point ``p`` and direction ``b`` pair. The value `y'` is calculated using the function:

        .. math::

            y'=-\\left(\\frac{b_xb_y}{1+b_z}\\right)p_x+\\left(1-\\frac{b^2_y}{1+b_z}\\right)p_y-b_yp_z

        :param p: The input point in 3D.
        :type p: list | tuple | np.ndarray

        :param b: The input direction vector in 3D.
        :type b: list | tuple | np.ndarray
        """

        yp = - b[0] * b[1] / (1 + b[2]) * p[0] + (1 - (b[1] ** 2 / (1 + b[2]))) * p[1] - (b[1] * p[2])
        return yp

    @staticmethod
    def _get_b(p1, p2) -> np.ndarray:

        """
        Get the direction vector between two points ``p1`` and ``p2`` (pointing in positive `z` direction).

        :param p1: Input point 1 in 3D.
        :type p1: list | tuple | np.ndarray

        :param p2: Input point 2 in 3D.
        :type p2: list | tuple | np.ndarray
        """

        b = p2 - p1 if (p2[2] > p1[2]) else p1 - p2
        b_norm = b / np.linalg.norm(b)

        return b_norm
