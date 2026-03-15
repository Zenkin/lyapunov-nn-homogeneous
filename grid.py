import numpy as np
import torch
from typing import Tuple
from numpy.typing import NDArray


def make_grid(
        x1_min: float = -4.0,  # минимальное значение по оси x1
        x1_max: float = 4.0,  # максимальное значение по оси x1
        x2_min: float = -4.0,  # минимальное значение по оси x2
        x2_max: float = 4.0,  # максимальное значение по оси x2
        grid: int = 101,  # количество точек по каждой оси
) -> Tuple[
    NDArray[np.float64],  # x1: одномерный массив координат по оси x1, форма (grid,)
    NDArray[np.float64],  # x2: одномерный массив координат по оси x2, форма (grid,)
    NDArray[np.float64],  # X1: матрица координат x1 на всей сетке, форма (grid, grid)
    NDArray[np.float64],  # X2: матрица координат x2 на всей сетке, форма (grid, grid)
    NDArray[np.float64],  # pts: список всех точек плоскости, форма (grid*grid, 2)
]:
    # Проверка: сетка должна содержать минимум 2 точки,
    # иначе невозможно построить интервал
    if grid < 2:
        raise ValueError(f"grid must be >= 2, got {grid}")

    # Проверка корректности диапазона по x1
    # верхняя граница должна быть строго больше нижней
    if not (x1_max > x1_min):
        raise ValueError(
            f"x1_max must be > x1_min, got x1_min={x1_min}, x1_max={x1_max}"
        )

    # Проверка корректности диапазона по x2
    if not (x2_max > x2_min):
        raise ValueError(
            f"x2_max must be > x2_min, got x2_min={x2_min}, x2_max={x2_max}"
        )

    # Формируем равномерную дискретизацию оси x1:
    # grid точек от x1_min до x1_max включительно
    x1 = np.linspace(
        float(x1_min),
        float(x1_max),
        int(grid),
        dtype=np.float64
    )

    # Аналогично формируем ось x2
    x2 = np.linspace(
        float(x2_min),
        float(x2_max),
        int(grid),
        dtype=np.float64
    )

    # Строим двумерную сетку координат.
    # X1[i, j] = x1[j]
    # X2[i, j] = x2[i]
    # indexing="xy" означает стандартную геометрию:
    # столбцы соответствуют x1, строки — x2
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")  # форма (grid, grid)

    # Преобразуем двумерную сетку в список точек.
    # reshape(-1) "сплющивает" матрицу в вектор.
    # stack(..., axis=1) объединяет координаты в пары (x1, x2).
    # В итоге получаем массив формы (N, 2),
    # где N = grid * grid — количество всех точек сетки.
    pts = np.stack(
        [X1.reshape(-1), X2.reshape(-1)],
        axis=1
    )

    # Возвращаем:
    # x1, x2  — оси
    # X1, X2  — матричную сетку для визуализации
    # pts     — список точек для вычислений (нейросети, f(x), градиенты)
    return x1, x2, X1, X2, pts

def sample_r_sphere_r12(
        n_samples: int,
        *,
        device=None,
        dtype=None,
) -> torch.Tensor:
    """
    Генерация точек на гомогенной сфере S_r(1) для весов r=(1,2):

        S_r(1) = { y in R^2 : |y1| + sqrt(|y2|) = 1 }

    Используется простой аналитический sampler.
    Берём s из [0, 1], затем задаём:

        |y1| = 1 - s
        |y2| = s^2

    После этого случайно выбираем знаки координат y1 и y2.
    Тогда автоматически выполняется:

        |y1| + sqrt(|y2|) = (1 - s) + s = 1

    Parameters
    ----------
    n_samples : int
        Количество генерируемых точек.
    device :
        Устройство torch (cpu / cuda). Если None, используется поведение torch по умолчанию.
    dtype :
        Тип данных torch. Если None, используется torch.get_default_dtype().

    Returns
    -------
    y : torch.Tensor
        Тензор формы (n_samples, 2) с точками на сфере S_r(1).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    if dtype is None:
        dtype = torch.get_default_dtype()

    # Равномерно сэмплируем параметр s in [0, 1]
    s = torch.rand(int(n_samples), device=device, dtype=dtype)

    # Задаём модули координат так, чтобы гарантировать условие сферы:
    # |y1| + sqrt(|y2|) = 1
    abs_y1 = 1.0 - s
    abs_y2 = s * s

    # Случайным образом назначаем знаки каждой координате.
    # True  -> +1
    # False -> -1
    sign_y1 = torch.where(
        torch.rand(int(n_samples), device=device) < 0.5,
        torch.tensor(-1.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )
    sign_y2 = torch.where(
        torch.rand(int(n_samples), device=device) < 0.5,
        torch.tensor(-1.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )

    y1 = sign_y1 * abs_y1
    y2 = sign_y2 * abs_y2

    y = torch.stack([y1, y2], dim=1)
    return y