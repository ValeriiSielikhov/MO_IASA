import numpy as np


def dichotomy(f, a, b, delta, tol=0.00001, max_iter=100):
    for _ in range(max_iter):
        d = (b - a) / 2 * delta
        x1 = (a + b) / 2 - d
        x2 = (a + b) / 2 + d
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        if abs(b - a) < 2 * tol:
            break
    x = (a + b) / 2
    return x


def conj_dir_opt(A, b):
    """
    Use conjugate gradients method to minimize
    1/2 * (Ax , x) + (b, x)
    A is symmetric non - negative defined n*n matrix ,
    b is n- dimensional vector
    """

    def target(x):
        return 1 / 2 * np.dot(A @ x, x) + np.dot(b, x)

    def grad(x):
        return A @ x + b

    # x = np.random.randn(len(b))
    x = np.array([100, -1040, 123])
    history = []
    history.append([*x])
    r, h = - grad(x), - grad(x)
    for _ in range(1, len(b) + 1):
        alpha = np.linalg.norm(r) ** 2 / np.dot(A @ h, h)
        x = x + alpha * h
        history.append([*x])
        beta = np.linalg.norm(r - alpha * (A @ h)) ** 2 / np.linalg.norm(r) ** 2
        r = r - alpha * (A @ h)
        h = r + beta * h
    return np.array(history)
 