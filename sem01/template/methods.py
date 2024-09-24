from typing import Callable
import numpy as np


F = Callable[[float, float], float]  # Функция двух переменных F(x, y)
Array = np.ndarray[np.float32]


def Euler(f: F, x0: float, y0: float, n: int = 1000, dx: float = 1e-2) -> Array:
    """
    Метод Эйлера.
    Порядок точности O(dx)
    """
    points = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        y += dx * f(x, y)
        x += dx
        points.append((x, y))
    return np.array(points)


def RungeKuttas2(f: F, x0: float, y0: float, n: int = 1000, dx: float = 1e-2) -> Array:
    """
    Метод Рунге-Кутты 2го порядка точности(метод Эйлера с пересчетом).
    Порядок точности O(dx^2)
    """
    points = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        y_hat = y + dx * f(x, y)
        y += dx * (f(x, y) + f(x + dx, y_hat))/2
        x += dx
        points.append((x, y))
    return np.array(points)


def RungeKuttas2(f: F, x0: float, y0: float, n: int = 1000, dx: float = 1e-2) -> Array:
    """
    Метод Рунге-Кутты 2го порядка точности(метод Эйлера с пересчетом).
    Порядок точности O(dx^2)
    """
    points = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        y_hat = y + dx * f(x, y)
        y += dx * (f(x, y) + f(x + dx, y_hat))/2
        x += dx
        points.append((x, y))
    return np.array(points)


def RungeKuttas4(f: F, x0: float, y0: float, n: int = 1000, dx: float = 1e-2) -> Array:
    """
    Метод Рунге-Кутты 4го порядка точности.
    Порядок точности: O(dx^4)
    """
    points = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + dx/2, y + k1*dx/2)
        k3 = f(x + dx/2, y + k2*dx/2)
        k4 = f(x + dx, y + k3*dx)
        y += dx * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += dx
        points.append((x, y))
    return np.array(points)


def AdamsBashfort(f: F, x0: float, y0: float, n: int = 1000, dx: float = 1e-2) -> Array:
    """
    Метод Адамса-Башфорта.
    Порядок точности: O(dx)
    """
    points = [(x0, y0), (x0 + dx, y0 + dx*f(x0, y0))]  # Метод Эйлера
    x, y = points[-1]
    for _ in range(1, n):
        x_i, y_i = points[-1]
        y += (3 * f(x, y) - f(x_i, y_i)) * dx / 2 
        x += dx
        points.append((x, y))
    return np.array(points)


def AdamsMultons(f: F, x0: float, y0: float, coef: list[float], n: int = 1000, dx: float = 1e-2) -> Array:
    """
    Метод Адамса-Мультона.
    Порядок точности: O(dx^k), гдe k = len(coef)
    """
    k = len(coef)
    points = [(x0, y0)]
    x, y = x0, y0
    # Первые шаги считаем через RK4
    for _ in range(k-1):
        k1 = f(x, y)
        k2 = f(x + dx/2, y + k1*dx/2)
        k3 = f(x + dx/2, y + k2*dx/2)
        k4 = f(x + dx, y + k3*dx)
        y += dx * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += dx
        points.append((x, y))

    for _ in range(k, n):
        y_hat = coef[0] * f(x, y)
        for i in range(1, k):
            x_i, y_i = points[-i]
            y_hat += coef[i] * f(x_i, y_i)
        y += y_hat * dx
        x += dx
        points.append((x, y))
    return np.array(points)


