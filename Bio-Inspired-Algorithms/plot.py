import operator

import matplotlib.pyplot as plt


def plot(points, path: list):
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, 'co')

    for _ in range(1, len(path)):
        i = path[_ - 1]
        j = path[_]
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i],
                  color='r', length_includes_head=True)

    # Close the loop and highlight the last and first point path
    i = path[-1]  # Last point index
    j = path[0]  # First point index
    plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i],
              color='b', length_includes_head=True)

    zoom_factor = 0.01
    plt.xlim(min(x) - zoom_factor, max(x) + zoom_factor)
    plt.ylim(min(y) - zoom_factor, max(y) + zoom_factor)
    plt.title("ACO Path By Travel Time")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
