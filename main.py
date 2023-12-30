from reconstruct import Hough
import numpy as np
import random


def main():
    v = np.arange(-10, 10, 1)
    point = [np.array(random.sample(range(0, 1000), 3)) for _ in range(10)]
    direction = [np.array(random.sample(range(0, 100), 3)) for _ in range(10)]

    points = []
    for i, p in enumerate(point):
        line = [p + direction[i] * n for n in v]
        for a in line:
            points.append(a)

    points = np.array(points) / 10

    hough = Hough(points, 1, 11, 1, 26)

    hough.increment_accumulator()
    hough.plot_accumulator()


if __name__ == "__main__":
    main()
