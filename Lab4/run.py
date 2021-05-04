import numpy as np
from Optimizer import conj_dir_opt
import matplotlib.pyplot as plt

A = 2 * np.array([[2, 0, 0.0005], [0, 8, 0], [0.0005, 0, 3]])
b = np.array([1, -1, 0])


def f(x_vect):
    x, y = x_vect[0], x_vect[1]
    x = np.array([x, y])
    return (1 / 2 * x.T @ A @ x + b.T @ x)[0, 0]



hist = conj_dir_opt(A, b)

print(f"{hist}")


def plot_trace(history, r = 1, rotate = None):
    u = np.linspace(0, np.pi, 10)
    v = np.linspace(0, 2 * np.pi, 10)

    x = r*np.outer(np.sin(u), np.sin(v))
    y = r*np.outer(np.sin(u), np.cos(v))
    z = r*np.outer(np.cos(u), np.ones_like(v))


    fig = plt.figure(figsize = (25, 25))
    ax = plt.axes(projection='3d')


    ax.plot(history.T[0], history.T[1], history.T[2], color = "red", marker = "o")
    if rotate is not None:
        ax.view_init(*rotate)


plot_trace(hist, rotate = (30, 150))
