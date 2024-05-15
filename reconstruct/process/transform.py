import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
from reconstruct.render import Plot3D
import time
from functools import lru_cache


from .geometry import FitHyperPlane
from reconstruct import constants
from reconstruct.data import Data, PointTree, detector


class Finder:
    
    def __init__(self, sim_data) -> None:

        self._points = sim_data.data

        # initialize a Data object which holds information about the detector and the point cloud
        self.data = Data(self._points)

        self._sim_data = sim_data

        # primary and secondary cone angles
        self._cone_angle = {
            "primary": 80 * np.pi / 180,
            "secondary": 20 * np.pi / 180
        }

        # doesnt do anything for now
        self._min_points = 5

        # maximum valid rmse for a track
        self._max_rmse = 10

        # store a record of previously run points
        self._run_points = {
            t: [] for t in ["ml", "dl", "vl"]
        }

    @lru_cache
    def _get_radius(self, i_root_layer, i_target_layer, cone_angle) -> float:

        target_layer = self.data.layer_from_index("ml", i_target_layer)
        base_layer = self.data.layer_from_index("ml", i_root_layer)

        radius: float = ((target_layer - base_layer) * np.tan(cone_angle))

        return radius

    @staticmethod
    def _project_point_from_root(root_point, point, layer) -> np.ndarray:

        if np.allclose(root_point, point):
            b = point
        else:
            b = point - root_point

        a = (layer - point[2]) / b[2] if not np.isclose(b[2], 0) and not np.allclose(root_point, point) else 0
        projected_point = point + a * b
        projected_point[2] = layer

        return projected_point

    @staticmethod
    def _is_valid_detection(root_point, projected_point, query_point, dim, verbose) -> bool:

        # subtract 0.02 m from the z component to allow for possible spread of detections within the tracker
        distance = np.sqrt(
            sum([
                np.square((query_point[i] - root_point[i]) / 100 - 0.02 if i == 2 else 0) for i in range(3)
            ])
        )

        min_dt = distance / constants.c * 1e9
        max_dt = 6 * min_dt

        query_time = query_point[3]
        if dim == 4:
            projected_time = projected_point[3]
            if verbose:
                print(f"Projected t = {projected_time}, Query t = {query_time}")
            if np.isclose(query_time, projected_time, atol=6):
                return True
            else:
                if verbose:
                    print("Not within projected time window.\n")
        else:
            root_time = root_point[3]
            dt = np.abs(query_time - root_time)
            if verbose:
                print(f"Root t = {root_time}, dt = {dt}")
            if max_dt > dt > min_dt:
                return True
            else:
                if verbose:
                    print(f"Not within physical constraints (min_dt = {min_dt}, max_dt = {max_dt}).\n")

        return False

    @lru_cache
    def _get_search_params(self, _primary_run, i_layer, i_base_layer) -> tuple[int, KDTree, float]:

        dimension = 3 if _primary_run else 4

        tree: KDTree = self.data.query_tree[dimension]["ml"][i_layer]
        radius = self._get_radius(
            i_base_layer, i_layer, self._cone_angle[
                "primary" if _primary_run else "secondary"
            ]
        )

        return dimension, tree, radius

    def _recurse_branch(self, branch: PointTree, key_path: list, verbose=False) -> PointTree:

        """Recursively populates a branch through the ml tracker from a root point on a defined primary layer."""

        if verbose:
            print(f"\nParams:\n\tBranch = {branch}\n\tKey path = {key_path}\n")

        # if the current recursion depth is greater than one (len(key_path) > 1), and if the last key is the same as its
        # previous key, the current search depth is 2 (we are attempting to skip a layer), otherwise it is 1
        depth = 1
        if len(key_path) > 1:
            if key_path[-1] == key_path[-2]:
                depth = 2

        # get the root point on the primary layer
        i_root_point, i_root_layer = list(branch.keys())[0]
        if verbose:
            print(f"Root: ({i_root_point, i_root_layer}), {self.data.point_from_index('ml', i_root_layer, i_root_point)}")

        # get the base point to work from (can be different or the same as root point)
        # if this is the first recursion level, key_path will be the root point and the base point is just the root
        i_base_point, i_base_layer = key_path[-1]
        if verbose:
            print(f"Base: ({i_base_point, i_base_layer}), {self.data.point_from_index('ml', i_base_layer, i_base_point)}")

        # get the previous base point (back point) for line projections
        if i_base_layer == i_root_layer:
            # for the first recursion loop, back_point = base_point = root_point
            i_back_point, i_back_layer = key_path[-1]
        else:
            # if not on the first recursion loop, back_point should be one previous to the base_point
            i_back_point, i_back_layer = key_path[-2]
            if i_base_layer == i_back_layer and len(key_path) > 2:
                # if not on the first or second recursion loop and base_point = back_point,
                # then back_point should be shifted to two previous to the base_point
                i_back_point, i_back_layer = key_path[-3]

        back_point = self.data.point_from_index("ml", i_back_layer, i_back_point)

        if verbose:
            print(f"Back: ({i_back_point, i_back_layer}), {back_point}\n")
            print(f"Depth = {depth}")

        # init primary run flag
        __primary = False
        if i_base_layer == i_root_layer:
            __primary = True

        if verbose:
            print(f"Primary = {__primary}")

        # get the active parent point
        base_point = self.data.point_from_index("ml", i_base_layer, i_base_point)

        # define the layer to search on
        i_search_layer = i_base_layer + depth
        search_layer = self.data.layer_from_index("ml", i_search_layer)

        if verbose:
            print(f"Search layer = {i_search_layer}")

        # retrieve the search parameters and tree
        dimension, tree, radius = self._get_search_params(__primary, i_search_layer, i_base_layer)

        if verbose:
            print(f"\nQuery params:\n\tDimension = {dimension}\n\tRadius = {radius}\n")

        # project the point from the back_point, thru the base_point, up to an intersection on the search layer
        projected_point = self._project_point_from_root(back_point, base_point, search_layer)

        if verbose:
            print(f"Projected point: {projected_point}\n")

        # query the tree for the current layer within the defined radius at the projected point
        query = tree.query_radius([projected_point[0: dimension]], r=radius)[0]

        if verbose:
            print(f"Query result: {query}\n")

        for q in query:

            if verbose:
                print(f"Checking point: {q, i_search_layer}, {self.data.point_from_index('ml', i_search_layer, q)}")

            query_point = self.data.point_from_index("ml", i_search_layer, q)
            if not self._is_valid_detection(base_point, projected_point, query_point, dimension, verbose):
                continue

            branch.append(key_path, [(q, i_search_layer)])

            if verbose:
                print(f"Confirmed point.\n")

            # if the search layer is below the highest possible layer, continue recursion
            # if it is equal to the maximum index, break out of recursion
            if not i_search_layer < len(detector.tracker["ml"]["z"]) - 1:
                continue

            if verbose:
                print(f"Running recursively.")
            branch = self._recurse_branch(branch, key_path + [(q, i_search_layer)], verbose)

        # if there are no duplicate indices (no layers have been skipped on this key path), the search layer is lower
        # than the highest possible layer, and the root layer is the lowest possible, attempt to find tracks
        # that skip the following layer
        if (
            len(set(key_path)) == len(key_path)
            and i_search_layer < len(detector.tracker["ml"]["z"]) - 1
            and i_root_layer == 0
        ):
            if verbose:
                print(f"Checking for layer skips.")
            branch.append(key_path, [(i_base_point, i_base_layer)])
            branch = self._recurse_branch(branch, key_path + [(i_base_point, i_base_layer)], verbose)

        return branch

    def _initialize_branch_recursion(self, tracks: dict, i_root_point, i_primary_layer) -> None:

        verbose = False
        # if (i_root_point, i_primary_layer) == (115, 0):
        #     verbose = True

        self_verbose = False
        if self_verbose:
            print(f"Root: ({i_root_point, i_primary_layer}), {self.data.point_from_index('ml', i_primary_layer, i_root_point)}")

        root_entry = (i_root_point, i_primary_layer)
        branch = PointTree(root_entry)
        filled_branch = self._recurse_branch(branch, [root_entry], verbose)

        track_set = []
        i_track_set = []
        for track in filled_branch.tracks():
            if not track:
                continue

            track_set.append([
                self.data.point_from_index("ml", lyr, pt) for pt, lyr in track
            ])
            i_track_set.append(track)

        if track_set:
            tracks["point"].append(track_set)
            tracks["index"].append(i_track_set)

    def find_tracks(self, i_primary_layer) -> dict:

        print(f"\n[ Branching from Layer {i_primary_layer}... ]\n")
        time.sleep(0.1)

        ml_points = self.data.sorted_points["ml"]["z"]
        tracks = {"point": [], "index": []}

        _iter_1 = tqdm(ml_points[i_primary_layer])
        for i_root_point, _ in enumerate(_iter_1):
            if (i_root_point, i_primary_layer) in self._run_points["ml"]:
                continue

            self._initialize_branch_recursion(tracks, i_root_point, i_primary_layer)

        return tracks

    def plot(self, line_params, save=False):

        plotter = Plot3D()

        plotter.points(self._points)
        plotter.lines(line_params, [6500, 28000], 6500)

        if save:
            plotter.save("track_fit.html")

        plotter.show()

    def run(self) -> None:

        def _track_valid(e):
            if e < self._max_rmse:
                return True
            return False

        for d in [3, 4]:
            self.data.generate_query_trees(d)

        line_params = []
        for primary_layer in range(2):

            # this needs to be tweaked a bit when implementing multiprocessing
            __track_dict = self.find_tracks(primary_layer)

            _iter = tqdm(enumerate(__track_dict["point"]))
            for i, tracks in _iter:

                refinery = Refinery(tracks)
                rmse, track, lps = refinery.minimize_rmse()

                if not _track_valid(rmse):
                    continue

                line_params.append(lps)
                for t in __track_dict["index"][i]:
                    for p in t:
                        self._run_points["ml"].append(p)

        detections = len(line_params)

        print(f"\nFound {detections} likely tracks.\n")
        time.sleep(0.1)

        self.plot(line_params, True)


class Refinery:
    """Houses the line detector."""

    def __init__(self, tracks: list) -> None:
        self._tracks = tracks

    def minimize_rmse(self) -> tuple[float, np.ndarray, np.ndarray]:

        rmse = []
        pb = []

        for track in self._tracks:
            line = FitHyperPlane(np.array(track))
            rmse.append(line.rmse())
            pb.append(line.params())

        p_min_rmse = min(range(len(rmse)), key=rmse.__getitem__)

        return rmse[p_min_rmse], self._tracks[p_min_rmse], np.array(pb[p_min_rmse])
