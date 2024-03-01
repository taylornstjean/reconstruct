from reconstruct import Transform
from testing import random_lines
from reconstruct.loader import SimData
from reconstruct.renderer import plot
import numpy as np


#  -------------------  MAIN DRIVER  -------------------  #


def test_transform(plate_spacing, pos_resolution, line_count, rerun=False):

    if not rerun:
        # generate new points
        points = random_lines(line_count, plate_spacing, pos_resolution)
    else:
        # load last run points
        points = np.loadtxt("points.txt")

    hough = Transform(points, 5, line_count + 5, plate_spacing, pos_resolution, 0.1 * np.pi / 180, 0.02, plot=True)
    hough.find_lines()


def run_transform(points, plate_spacing, pos_resolution):

    hough = Transform(points, 5, len(points), plate_spacing, pos_resolution, 0.1 * np.pi / 180, 0.02, plot=True)
    hough.find_lines()


def main():

    # test_transform(line_count=10, pos_resolution=0.01, plate_spacing=0.8, rerun=False)

    items = ["Hit_x", "Hit_z", "Hit_y", "Hit_time"]
    path = "./data/muons_35pT/20231204/203838/run0.root"  # "./data/muons_27pT/20231116/051933/run0.root"
    sim_data = SimData(path, items)

    plot(sim_data.data)

    # run_transform(points=sim_data.data, pos_resolution=10, plate_spacing=800)


if __name__ == "__main__":
    main()
