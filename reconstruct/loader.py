import uproot as u
from uproot.models.TTree import Model_TTree_v20
import pandas as pd
import numpy as np
from tqdm import tqdm


class SimData:

    def __init__(self, path, items, **kwargs) -> None:

        """
        Load a ROOT file containing simulated data.

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

    @property
    def data(self) -> np.ndarray:

        """Retrieve the data from the file."""

        file: Model_TTree_v20 = self._load_file()
        data: list = file.arrays(self._items).to_list()

        return self._parse_data(data)
    
    def _load_file(self) -> Model_TTree_v20:

        """Load the file into memory."""

        file: Model_TTree_v20 = u.open("{}:{}".format(self._path, self._branch))
        return file
    
    @staticmethod
    def _parse_data(data) -> np.ndarray:

        """Parse the data in the file and select the desired items."""

        frames: list[pd.DataFrame] = [pd.DataFrame(stack) for stack in data]
        df: pd.DataFrame = pd.concat(frames)
        array: np.ndarray = df.to_numpy()

        points: list = array.tolist()
        _ignore: list = []
        _to_delete: list = []

        _iter = tqdm(array)

        for i, point in enumerate(_iter):
            _ignore.append(i)

            for j, com_point in enumerate(array):
                if j in _ignore:
                    continue

                if np.allclose(point, com_point, atol=10):
                    _ignore.append(j)
                    _to_delete.append(j)

        for i in sorted(_to_delete, reverse=True):
            points.pop(i)

        print(points)
        print(_to_delete)

        return np.array(points)
