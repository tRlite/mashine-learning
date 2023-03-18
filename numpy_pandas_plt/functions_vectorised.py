import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    X = np.diag(X)
    if np.sum(X >= 0) == 0:
        return -1
    X = X[X >= 0]
    return np.sum(X)


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x, y = np.sort(x), np.sort(y)
    return not np.any(x-y)


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    x = x[1:] * x[:-1]
    return -1 if np.all(x % 3) else np.max(x[x % 3 == 0]) 


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.sum(image*weights, axis = 2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x = np.repeat(x[:,0], x[:,1])
    y = np.repeat(y[:,0], y[:,1])
    if len(x) != len(y):
        return -1 
    return np.sum(x * y)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    scalar = np.einsum("ik, jk-> ij", X, Y)
    mod_x = np.sum(X*X, axis = 1)[:,np.newaxis] ** 0.5
    mod_y = np.sum(Y*Y, axis = 1)[np.newaxis:,] ** 0.5
    mod = mod_x * mod_y
    mask = mod == 0
    mask2 = mod != 0
    scalar[mask] = 1
    scalar[mask2] = scalar[mask2] / mod[mask2]
    return scalar
