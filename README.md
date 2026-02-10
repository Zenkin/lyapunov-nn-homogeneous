# lyapnn — Unified Lyapunov pipeline

This project now runs one end-to-end workflow:

1) **Train V_inf on f_inf** over a box `Omega` (no shift).
2) **Diagnostics for V_inf** on `Omega`.
3) **Compute V_full** using V_inf and f_full on `Omega`, then plot diagnostics.
4) **Train W** on a local box around equilibrium (shifted coordinates).
5) **Diagnostics for W** on that local box.
6) **Blend V_full/W** on `Omega` using:
   - outside `W_box`: V_full
   - inside `X_box`: W
   - between: `max(V_full, W)` with corresponding derivative
7) **Final diagnostics** on `Omega`.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run (single command)
```bash
lyapnn --outdir runs/output \
  --omega_x1_min -20 --omega_x1_max 20 --omega_x2_min -20 --omega_x2_max 20 \
  --w_box_x1_min -5 --w_box_x1_max 5 --w_box_x2_min -5 --w_box_x2_max 5 \
  --x_box_x1_min -1 --x_box_x1_max 1 --x_box_x2_min -1 --x_box_x2_max 1
```

Notes:
- `Omega` is in **original coordinates** (no shift).
- `W_box` and `X_box` are in **shifted coordinates** (centered at x_eq).
- Heatmaps are saved as a single image with V and dV side-by-side; bad regions highlight `V<=0` or `dV>=0`.
- 3D plots are shown sequentially so you can rotate them.
- In addition to PNG plots, all numeric data used for the plots is saved in NumPy format:
  - per stage: `vinf/vinf_heatmaps.npz`, `vfull/vfull_heatmaps.npz`, `w/w_heatmaps.npz`, `final/v_final_heatmaps.npz`
  - aggregated: `all_plot_data.npz` in the run root directory.
