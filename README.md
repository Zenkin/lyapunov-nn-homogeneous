# lyapnn — Homogeneous NN Lyapunov (r=(1,2))

This repo contains two training stages and matching visual diagnostics:

- **Step 2**: train a homogeneous NN Lyapunov candidate `V` on the infinity approximation `f_inf` over `S_r(1)`.
- **Step 3**: train a near-zero corrector `W(x)=||T(x)||^2` on a user-defined rectangle in *shifted* coordinates.

## Install (editable)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -U pip
pip install -e .
```

## Train (Step 2)
```bash
lyapnn step2-train --device cpu --steps 1000 --outdir runs/step2
```

## Plot diagnostics (heatmaps + 3D surfaces)
```bash
lyapnn step2-plot --ckpt runs/step2/step2_V.pt --outdir runs/step2
```

## Train + plot (Step 3)
```bash
lyapnn step3-train --outdir runs/step3 \
  --x1_min 0 --x1_max 10 --x2_min -10 --x2_max 2 \
  --steps 50000 --batch 2048 --lr 1e-3

lyapnn step3-plot --ckpt runs/step3/W_model.pt --outdir runs/step3 \
  --x1_min 0 --x1_max 10 --x2_min -10 --x2_max 2 --plot_3d --save
```

Outputs are written to `runs/step2/` and `runs/step3/` by default.

Notes:
- No seaborn; matplotlib only.
- `V(x) = rho(x)^mu * W(y)` with `W(y) = softplus(Wraw(y)) + eps`.

### Backward-compatible scripts
The `scripts/` folder still exists, but now it simply forwards to the unified CLI.
For open-source usage, prefer `lyapnn ...` commands.
