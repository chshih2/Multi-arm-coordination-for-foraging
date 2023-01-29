import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt


def cartesian2polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return [rho, phi]


def y_rotation(vector, theta):
    theta = theta / 180 * np.pi
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector.T).T


def read_polygon_workspace(file_dir, view=False):
    hull_pts = np.load(file_dir + "/0528_concave_hull_points.npy")
    polygon = Polygon(tuple(zip(hull_pts[0], hull_pts[1])))
    if view:
        view_pts = hull_pts.copy()
        view_pts[:, 2:-1] *= 2
        view_polygon = Polygon(tuple(zip(view_pts[0], view_pts[1])))
        return hull_pts, polygon, view_pts, view_polygon
    return hull_pts, polygon


def generate_targets_inside_workspace(polygon):
    outside_flag = True
    while outside_flag:
        target = np.zeros(3)
        target[0] = np.random.rand() * 1.85 - 0.6
        target[1] = np.random.rand() * 1.25
        outside_flag = not check_point_in_polygon(target, polygon)
    return target


def check_point_in_polygon(x, polygon):
    point = Point(x[0], x[1])
    return polygon.contains(point)


def check_point_in_polygon_with_boundary(x, polygon):
    point = Point(x[0], x[1])
    return polygon.buffer(1e-6).intersects(point)


def plot_polygon_workspace(hull_pts, x_shift=0.0, ax=None):
    if ax == None:
        plt.plot(hull_pts[0] + x_shift, hull_pts[1], color='black', alpha=0.1)
    else:
        plot1, = ax.plot(hull_pts[0] + x_shift, hull_pts[1], color='black', alpha=0.1, linewidth=10)
        return plot1
