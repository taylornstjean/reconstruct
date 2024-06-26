import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
import time

from reconstruct.data import detector


def _float_multiply(n: float, m: float | int):
    """Accurately multiply floats with NumPy, because Python doesn't know how to do math."""
    result = int(np.round(m * np.float128(n), 0))
    return result


class Data:

    def __init__(self, points) -> None:
        self._points = points

        # sort points by layer
        self._sorted_points = self._sort_by_layer()

        # initialize a data tree (allow for timing dependent and timing independent queries)
        self._query_tree: dict[int: dict[str: dict[float: KDTree]]] = {
            d: {"ml": {}, "dl": {}, "vl": {}} for d in [3, 4]
        }

    @property
    def query_tree(self):
        return self._query_tree

    @property
    def sorted_points(self):
        return self._sorted_points

    @staticmethod
    def layers(tracker, axis):
        return list(detector.tracker[tracker][axis].values())

    def _sort_by_layer(self) -> dict[str: dict[str: dict[int: list[np.ndarray]]]]:
        """Sort the data by which layer the detection occurred on."""

        ml_tracker: dict[str: dict] = {
            "z": {_float_multiply(z, 10): [] for z in self.layers("ml", "z")}
        }  # multi layer tracker detections

        dl_tracker: dict[str: dict] = {
            "z": {_float_multiply(z, 10): [] for z in self.layers("dl", "z")}
        }  # double layer tracker detections

        vl_tracker: dict[str: dict] = {
            "z": {_float_multiply(z, 10): [] for z in self.layers("vl", "z")},
            "y": {_float_multiply(y, 10): [] for y in self.layers("vl", "y")}
        }  # veto/wall layer tracker detections

        print("\n[ Sorting by Tracker... ]\n")

        _iter = tqdm(self._points)

        # sort each point by detection layer
        for point in _iter:
            if point[1] <= detector.wall_max_y:
                try:
                    vl_tracker["y"][_float_multiply(point[1], 10)].append(point)
                    continue
                except KeyError:
                    pass
            else:
                for tracker in [ml_tracker, dl_tracker, vl_tracker]:
                    try:
                        tracker["z"][_float_multiply(point[2], 10)].append(point)
                        continue
                    except KeyError:
                        pass

        # remap layer to layer indices

        ml_tracker_remap = {}
        for axis, layer_data in ml_tracker.items():
            ml_tracker_remap[axis] = {i_layer: data for i_layer, data in enumerate(layer_data.values())}

        dl_tracker_remap = {}
        for axis, layer_data in dl_tracker.items():
            dl_tracker_remap[axis] = {i_layer: data for i_layer, data in enumerate(layer_data.values())}

        vl_tracker_remap = {}
        for axis, layer_data in vl_tracker.items():
            vl_tracker_remap[axis] = {i_layer: data for i_layer, data in enumerate(layer_data.values())}

        return {"ml": ml_tracker_remap, "dl": dl_tracker_remap, "vl": vl_tracker_remap}

    def generate_query_trees(self, dim):

        print(f"\n[ Generating Query Tree (dim={dim})... ]\n")
        time.sleep(0.1)

        _iter = tqdm(self.layers("ml", "z"))
        for i_layer, _ in enumerate(_iter):
            self._query_tree[dim]["ml"][i_layer] = KDTree(
                np.array([p[0:dim] for p in self._sorted_points["ml"]["z"][i_layer]])
            )

    def point_from_index(self, tracker, i_layer, i_point) -> np.ndarray:

        points = self._sorted_points[tracker]["z"]
        point: np.ndarray = points[i_layer][i_point]

        return point

    def layer_from_index(self, tracker, i_layer) -> float:

        if tracker not in ["ml", "dl", "vl"]:
            raise ValueError

        layer = self.layers(tracker, "z")[i_layer]

        return layer
