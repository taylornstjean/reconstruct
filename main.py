from reconstruct import Hough
import numpy as np
import random


def main():
    v = np.arange(0, 4, 1)

    line_count = 5

    point = [np.array([np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), 0]) for _ in range(line_count)]
    direction = [np.array([np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), 100]) / 10 for _ in range(line_count)]

    points = []
    for i, p in enumerate(point):
        line = [p + direction[i] * n for n in v]
        for a in line:
            points.append(a)

    points = np.array(points) / 10

    hough = Hough(points, 1, line_count, 1, 10)

    hough.increment_accumulator()
    hough.plot_accumulator()


if __name__ == "__main__":
    main()
