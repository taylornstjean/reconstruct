import uproot as u
from uproot.models.TTree import Model_TTree_v20
import time
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
from tqdm import tqdm
import reconstruct.detector as detector


class Loader:

    def __init__(self, path, items, **kwargs) -> None:
        """Load a ROOT file containing simulated data.

        :param path: The path to the ROOT file (relative or absolute).
        :type path: str

        :param items: The objects to pull from the file (under branch specified by ``root_path``).
        :type items: list

        :Keyword Arguments:
            * *branch* (``str``) --
              The location of the branch within the ROOT file to find the items specified. Defaults to ``"box_run;1"``.
        """

        self._path: str = path
        self._items: list = items

        self._branch: str = kwargs.get("branch", "box_run;1")

        file: Model_TTree_v20 = self._load_file()
        data: list = file[self._branch].arrays(self._items).to_list()

        self._data = self._parse_data(data)
        self._metadata = self._load_file_metadata(file)

    @property
    def data(self) -> np.ndarray:
        """Retrieve the data from the file."""

        return self._data
    
    @property
    def metadata(self) -> dict:
        """Retrieve the metadata from the file."""

        return self._metadata
    
    def _load_file(self) -> Model_TTree_v20:
        """Load the file into memory."""

        file: Model_TTree_v20 = u.open(self._path)
        return file

    def _load_file_metadata(self, file: Model_TTree_v20) -> dict:
        """Load metadata from the file."""

        metadata = {v.members["fName"]: v.members["fTitle"] for k, v in file.items() if k != self._branch}

        return metadata

    def _parse_data(self, data) -> np.ndarray:
        """Parse the data in the file and select the desired items."""

        frames: list[pd.DataFrame] = [pd.DataFrame(stack) for stack in data]
        df: pd.DataFrame = pd.concat(frames)
        array: np.ndarray = df.to_numpy()

        flat_array = self._flatten(array)

        flat_condense_array = self._compress(flat_array)

        print("\nLoaded {} points.\n".format(len(flat_condense_array)))
        time.sleep(0.01)

        return flat_condense_array

    @staticmethod
    def _flatten(data) -> np.ndarray:
        """Project each layer to a 2D plane."""

        print("\n[ Flattening... ]\n")
        time.sleep(0.01)

        # horizontal tracking layers projected vertically
        _v_plate_projections: dict[tuple[float]: float] = {
            **detector.tracker["ml"]["z"],
            **detector.tracker["dl"]["z"],
            **detector.tracker["vl"]["z"]
        }

        # vertical tracking layers projected horizontally
        _h_plate_projections: dict[tuple[float]: float] = detector.tracker["vl"]["y"]

        def _project(projector: dict, p, i):
            for bounds, mean in projector.items():
                if bounds[0] <= p[i] <= bounds[1]:
                    p[i] = mean
                    return p.tolist()
                    
        points = []

        _iter = tqdm(data)
        
        for point in _iter:
            if point[1] > detector.wall_max_y:
                flattened_point = _project(_v_plate_projections, point, 2)
                if flattened_point:
                    points.append(flattened_point)
            else:
                flattened_point = _project(_h_plate_projections, point, 1)
                if flattened_point:
                    points.append(flattened_point)

        return np.array(points)
    
    @staticmethod
    def _compress(data) -> np.ndarray:
        """Remove redundant detections from energy deposits."""

        print("\n[ Compressing... ]\n")
        time.sleep(0.01)

        data_tree: KDTree = KDTree(data)

        neighbors: list[np.ndarray] = data_tree.query_radius(data, 5)

        points = []
        _ignore = set()

        _iter = tqdm(neighbors)

        for n in _iter:
            if any([i in _ignore for i in n]):
                continue

            p_select: list[np.ndarray] = data[n]
            p_mean = np.mean(p_select, axis=0).tolist()

            points.append(p_mean)
            _ignore.update(set(n))
        
        return np.array(points)
