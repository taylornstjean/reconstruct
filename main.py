from reconstruct import Transform
from testing import random_lines
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


def main():

    test_transform(line_count=100, pos_resolution=0.01, plate_spacing=0.8)


if __name__ == "__main__":
    main()
