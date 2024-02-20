import numpy as np

#  -------------------  FUNCTIONS FOR TESTING  -------------------  #


def random_lines(n, plate_spacing, pos_resolution):
    """Generate a set of ``n`` random lines with artificial noise.

    :param n: The number of lines to generate.
    :type n: int

    :param plate_spacing: The vertical spacing in meters between each tracker in the main detector.
    :type plate_spacing: int | float

    :param pos_resolution: The positional resolution in meters of the trackers in the detector.
    :type pos_resolution: int | float
    """

    v = np.arange(-2, 3, 1) * plate_spacing

    # randomly generate point and direction vectors
    point = [np.array([
        np.random.randint(-1000, 1000),
        np.random.randint(-1000, 1000),
        0,
        np.random.randint(0, 10000)
    ]) / 10 for _ in range(n)]

    direction = [np.array([
        np.random.randint(-1000, 1000),
        np.random.randint(-1000, 1000),
        100,
        267
    ]) / 100 for _ in range(n)]

    # skew the data to better match experimental conditions
    direction += np.array([14, 0, 0, 0])

    # generate line points given source and direction
    points = []
    for i, p in enumerate(point):
        line = [p + direction[i] * n for n in v]
        for a in line:
            points.append(a)

    # add noise to data to better match experimental conditions
    points = np.array(points)
    points += np.random.normal(size=points.shape) * [pos_resolution, pos_resolution, 0, 0]

    # save points in case of need to rerun
    np.savetxt("points.txt", points)

    return points
