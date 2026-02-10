# lyapnn — единый пайплайн для обучения и диагностики функций Ляпунова

`lyapnn` — это CLI-инструмент, который запускает полный workflow для системы Даффинга с трением:

1. обучение `V_inf` на однородной аппроксимации динамики `f_inf`;
2. диагностика `V_inf` и `dV_inf` на области `Omega`;
3. перенос `V_inf` в полную систему `f_full` (`V_full`, `dV_full`);
4. обучение локальной функции `W` вокруг равновесия;
5. диагностика `W` и `dW`;
6. сборка финальной функции `V_final` через blending `V_full` и `W`;
7. сохранение графиков и численных данных (`.npz`) для всех этапов.

---

## Требования

- Python `>= 3.9`;
- зависимости: `numpy`, `torch`, `matplotlib`.

Они автоматически ставятся из `pyproject.toml` при установке пакета.

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

После установки будет доступна команда `lyapnn`.

---

## Быстрый запуск

### 1) Запуск с дефолтными параметрами

```bash
lyapnn --outdir runs/default --no_show
```

`--no_show` отключает интерактивные окна matplotlib (удобно для SSH/CI).

### 2) Запуск с пользовательскими областями

```bash
lyapnn --outdir runs/custom --no_show \
  --omega_x1_min -20 --omega_x1_max 20 --omega_x2_min -20 --omega_x2_max 20 \
  --w_box_x1_min -5 --w_box_x1_max 5 --w_box_x2_min -5 --w_box_x2_max 5 \
  --x_box_x1_min -1 --x_box_x1_max 1 --x_box_x2_min -1 --x_box_x2_max 1
```

---

## Ключевые понятия по координатам

- `Omega` задается в **исходных координатах** системы: `(x1, x2)`.
- `W_box` и `X_box` задаются в **сдвинутых координатах**: `(x1_tilde, x2)`, где `x1_tilde = x1 - x_eq`.

Это важно для корректной интерпретации тепловых карт и масок blending-а.

---

## Как строится `V_final`

На всей `Omega` считается `V_full` и `W`, затем применяется правило:

- вне `W_box` используется `V_full`;
- внутри `X_box` используется `W`;
- в области `W_box \ X_box` берется `max(V_full, W)`;
- производная выбирается согласованно с выбранной функцией (`dV_full` или `dW`).

---

## Основные CLI-параметры

Полный список:

```bash
lyapnn --help
```

Ниже — самые важные группы параметров.

### Общие

- `--outdir` — директория с результатами (`runs/output` по умолчанию);
- `--device` — устройство для torch (`cpu`, `cuda`, ...);
- `--dtype` — `float32` или `float64`;
- `--seed` — random seed;
- `--no_show` — не показывать интерактивные графики;
- `--no_save` — не сохранять PNG-файлы (npz при этом сохраняются).

### Сетки и области

- `--grid` — размер сетки по каждой оси (например, `101` => `101x101`);
- `--omega_*` — границы основной области `Omega`;
- `--w_box_*` — границы области, где разрешено использовать `W`;
- `--x_box_*` — внутреннее ядро, где всегда берется `W`.

### Обучение `V_inf`

- `--vinf_mu`, `--vinf_alpha` — параметры модели `V_inf`;
- `--vinf_hidden`, `--vinf_depth` — архитектура;
- `--vinf_steps`, `--vinf_batch`, `--vinf_lr` — обучение;
- `--vinf_log_every` — частота логирования;
- `--vinf_normalize_margin` — нормализация margin (1/0).

### Обучение `W`

- `--w_hidden`, `--w_depth` — архитектура;
- `--w_steps`, `--w_batch`, `--w_lr` — обучение;
- `--w_log_every` — частота логирования;
- `--w_r_min`, `--w_margin`, `--w_alpha_pos`, `--w_eps_s`, `--w_lam_s` — коэффициенты и ограничения loss/регуляризации.

---

## Что сохраняется в `outdir`

Структура результатов:

```text
<outdir>/
  vinf/
    vinf.pt
    vinf_heatmaps.png
    vinf_3d.png
    dvinf_3d.png
    vinf_heatmaps.npz
  vfull/
    vfull_heatmaps.png
    vfull_3d.png
    dvfull_3d.png
    vfull_heatmaps.npz
  w/
    w_model.pt
    w_heatmaps.png
    w_3d.png
    dw_3d.png
    w_heatmaps.npz
  final/
    v_final_heatmaps.png
    v_final_3d.png
    dv_final_3d.png
    v_final_heatmaps.npz
  all_plot_data.npz
```

`all_plot_data.npz` содержит агрегированные массивы по всем этапам, включая итоговые маски blending-а (`final_w_mask`, `final_x_mask`).

---

## Запуск как Python-модуль

Если не хотите пользоваться entrypoint-скриптом, можно так:

```bash
python -m lyapnn.cli --outdir runs/module --no_show
```

---

## Практические советы

- Для первых прогонов уменьшайте число шагов (`--vinf_steps`, `--w_steps`) и сетку (`--grid`) для быстрой обратной связи.
- Если работаете без GUI (сервер, Docker, CI), почти всегда нужен `--no_show`.
- Для воспроизводимости фиксируйте `--seed` и сохраняйте команду запуска рядом с артефактами.
