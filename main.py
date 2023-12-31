from reconstruct import Hough
import numpy as np


def main():
    v = np.arange(-2, 2, 1)

    line_count = 100

    point = [np.array([
        np.random.randint(-1000, 1000),
        np.random.randint(-1000, 1000),
        0
    ]) / 10 for _ in range(line_count)]

    direction = [np.array([
        np.random.randint(-1000, 1000),
        np.random.randint(-1000, 1000),
        1000
    ]) / 100 for _ in range(line_count)]

    points = []
    for i, p in enumerate(point):
        line = [p + direction[i] * n for n in v]
        for a in line:
            points.append(a)

    points = np.array(points) / 10
    points += np.random.normal(size=points.shape) * [0.1, 0.1, 0]

    hough = Hough(points, 4, line_count, 0.5, 30)

    hough.increment_accumulator()
    hough.plot_accumulator()


if __name__ == "__main__":
    main()
