from reconstruct import Hough
import numpy as np
import random


def main():
    v = np.arange(0, 1, 0.1)
    point = np.array(random.sample(range(0, 1000), 3))
    direction = np.array(random.sample(range(0, 100), 3))
    points = np.array([point + direction * n for n in v]) / 10
    hough = Hough(points, 1, 1, 0.1)

    hough.increment_accumulator()
    hough.plot_accumulator()


if __name__ == "__main__":
    main()
