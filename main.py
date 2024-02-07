from reconstruct import Transform
import numpy as np
import matplotlib.pyplot as plt


def percent_success(b, xy, line_count):
    iterations = 10
    result = np.sum([run_hough(b, xy, line_count) / line_count for _ in range(iterations)]) / iterations * 100
    return result


def run_hough(btol, xytol, line_count):
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

    direction += np.array([14, 0, 0])

    line_params = [[p, d / np.linalg.norm(d)] for p, d in zip(point, direction)]

    points = []
    for i, p in enumerate(point):
        line = [p + direction[i] * n for n in v]
        for a in line:
            points.append(a)

    points = np.array(points)
    points += np.random.normal(size=points.shape) * [0.01, 0.01, 0]

    hough = Transform(points, 5, line_count + 5, btol, xytol, 0 * np.pi / 180, plot=True)

    lines = hough.find_lines()

    count = 0
    for line in lines:
        for lp in line_params:
            if np.allclose(line[0], lp[0], atol=0.04) and np.allclose(line[1], lp[1], atol=0.04):
                count += 1

    return count


def accuracy_check():

    results = {}
    for i in range(1, 40):
        results[i] = percent_success(line_count=i, xy=0.6, b=1 * np.pi / 180)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    for x, y in results.items():
        ax.scatter(x, y)

    plt.show()


def main():

    run_hough(line_count=50, xytol=0.1, btol=0.5 * np.pi / 180)


if __name__ == "__main__":
    main()
