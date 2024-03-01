import uproot as u
from uproot.models.TTree import Model_TTree_v20
import pandas as pd
import numpy as np


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
    
    def _parse_data(self, data) -> np.ndarray:

        """Parse the data in the file and select the desired items."""

        frames: list[pd.DataFrame] = [pd.DataFrame(stack) for stack in data]
        df: pd.DataFrame = pd.concat(frames)
        df.reset_index()
        array: np.ndarray = df.to_numpy()

        return array
