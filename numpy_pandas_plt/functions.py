from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """ 
    sum = 0
    flag = False
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            flag = True 
            sum += X[i][i]
    if flag:
        return sum
    return -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    if (len(x) != len(y)):
        return False
    x, y = sorted(x), sorted(y)
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True 


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    max_prod = None 
    for i in range(len(x) - 1):
        if not (x[i] * x[i+1]) % 3:
            if max_prod == None:
                max_prod = x[i]*x[i+1]
            else:
                max_prod = max(max_prod, x[i]*x[i+1])
    return max_prod if max_prod != None else -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res = []
    for i in image:
        row = []
        for j in i:
            sum = 0
            for k in range(len(j)):
                sum += j[k] * weights[k]
            row.append(sum)
        res.append(row)
    return res


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    size_x = 0
    size_y = 0
    for i in x:
        size_x += i[1]
    for i in y:
        size_y += i[1]
    if size_x != size_y:
        return -1
    res = 0
    i = 0
    j = 0 
    while i < len(x): 
        if x[i][1] >= y[j][1]:
            res += y[j][1] * x[i][0] * y[j][0]
            x[i][1] -= y[j][1]
            j += 1
            if x[i][1] == 0:
                i += 1
        else:
            res += x[i][1] * x[i][0] * y[j][0]
            y[j][1] -= x[i][1]
            i += 1
    return res


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    res = []
    for i in X:
        row = []
        for j in Y:
            scalar = 0
            mod_x = 0
            mod_y = 0 
            for k in range(len(j)):
                scalar += i[k] * j[k]
                mod_x += i[k] ** 2
                mod_y += j[k] ** 2
            mod_x **= 0.5
            mod_y **= 0.5
            if  mod_x > 0.00005 and mod_y > 0.00005:
                row.append(scalar/(mod_x*mod_y))
            else:
                row.append(1)
        res.append(row)
    return res
