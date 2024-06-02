'''
'''
import matplotlib.pyplot as plt
import numpy as np


def plot_history(targets, label=None, x_range=None, y_range=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(targets, label=label)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    if label is not None:
        ax.legend()
    return ax


def plot3d(x_range, y_range, z_function, grad_function):
    xs = np.linspace(x_range[0], x_range[1], 100)
    ys = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(xs, ys)
    points = np.c_[np.ravel(X), np.ravel(Y)]
    Z = np.array([z_function(x, y) for x, y in points])
    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(16, 8))
    subplot = fig.add_subplot(1, 2, 1, projection='3d')
    subplot.plot_surface(X, Y, Z, cmap=plt.cm.Blues_r)
    subplot = fig.add_subplot(1, 2, 2)
    subplot.contourf(X, Y, Z, levels=50,
                     cmap=plt.cm.Blues_r, alpha=0.5)

    xs = np.linspace(x_range[0], x_range[1], 15)
    ys = np.linspace(y_range[0], y_range[1], 15)
    X, Y = np.meshgrid(xs, ys)
    points = np.c_[np.ravel(X), np.ravel(Y)]
    G = [grad_function(x, y) for x, y in points]
    U = np.array([u for u, v in G])
    V = np.array([v for u, v in G])
    U = U.reshape(X.shape)
    V = V.reshape(X.shape)

    subplot.set_aspect('equal')
    subplot.quiver(X, Y, U, V, color='red',
                   scale=16, headlength=6, headwidth=7)
    subplot.set_position([0.5, 0.2, 0.25, 0.50])
    subplot.set_xticks([])
    subplot.set_yticks([])


def eye():
    plt.show()
    plt.close()
