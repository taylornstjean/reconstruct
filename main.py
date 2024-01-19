from reconstruct import Hough
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():

    line_count = 6

    def run_hough(btol, xytol):
        v = np.arange(-2, 3, 1) * 0.8

        point = [np.array([
            np.random.randint(-1000, 1000),
            np.random.randint(-1000, 1000),
            0
        ]) / 10 for _ in range(line_count)]

        direction = [np.array([
            np.random.randint(-1000, 1000),
            np.random.randint(-1000, 1000),
            100
        ]) / 100 for _ in range(line_count)]

        line_params = [[p, d / np.linalg.norm(d)] for p, d in zip(point, direction)]

        points = []
        for i, p in enumerate(point):
            line = [p + direction[i] * n for n in v]
            for a in line:
                points.append(a)

        points = np.array(points)
        points += np.random.normal(size=points.shape) * [0.01, 0.01, 0]

        hough = Hough(points, 4, line_count, btol, xytol)

        lines = hough.find_lines()

        count = 0
        for line in lines:
            for lp in line_params:
                if np.allclose(line[0], lp[0], atol=0.04) and np.allclose(line[1], lp[1], atol=0.04):
                    count += 1

        return count

    iterations = 100
    _iter = tqdm(range(iterations))
    percent_success = lambda b, xy: np.sum([run_hough(b, xy) / line_count for _ in _iter]) / iterations * 100

    print(percent_success(4 * np.pi / 180, 0.4))

    #results = {}
    #_iter = tqdm(np.arange(3 * np.pi / 180, 5 * np.pi / 180, 0.5 * np.pi / 180), desc="B Iter")
    #_iter2 = tqdm(np.arange(3, 6, 1), desc="XY Iter")
    #for b in _iter:
    #    for xy in _iter2:
    #        results[(b, xy)] = percent_success(b, xy)

    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(projection="3d")

    #for (x, y), v in results.items():
    #    ax.scatter(x, y, v)

    #plt.show()

    # run_hough(4 * np.pi / 180, 0.4)

    # best params btol=4 degrees, xytol=0.4 (with 3 lines)


if __name__ == "__main__":
    main()
