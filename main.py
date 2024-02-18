from reconstruct import Transform
import numpy as np


#  -------------------  MAIN DRIVER FOR TESTING  -------------------  #


def run_hough(btol, xytol, line_count):
    v = np.arange(-2, 3, 1) * 0.8

    point = [np.array([
        np.random.randint(-1000, 1000),
        np.random.randint(-1000, 1000),
        0,
        np.random.randint(0, 10000)
    ]) / 10 for _ in range(line_count)]

    direction = [np.array([
        np.random.randint(-1000, 1000),
        np.random.randint(-1000, 1000),
        100,
        267
    ]) / 100 for _ in range(line_count)]

    direction += np.array([14, 0, 0, 0])

    points = []
    for i, p in enumerate(point):
        line = [p + direction[i] * n for n in v]
        for a in line:
            points.append(a)

    points = np.array(points)
    points += np.random.normal(size=points.shape) * [0.01, 0.01, 0, 0]

    np.savetxt("points.txt", points)

    hough = Transform(points, 5, line_count + 5, btol, xytol, 0.1 * np.pi / 180, 0.02, plot=True)

    hough.find_lines()


def run_points(line_count, btol, xytol):

    points = np.loadtxt("points.txt")
    hough = Transform(points, 5, line_count + 5, btol, xytol, 0.1 * np.pi / 180, 0.02, plot=True)

    hough.find_lines()


def main():

    run_hough(line_count=50, xytol=0.3, btol=0.5 * np.pi / 180)


if __name__ == "__main__":
    main()
