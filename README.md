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

## Unified workflow (Step 2 → Step 3)
Run training and plotting in one command with consistent defaults (near-zero region defaults to [-5, 5]).
```bash
lyapnn workflow --device cpu --dtype float32 \
  --step2_steps 1000 --step3_steps 50000 \
  --x1_min -5 --x1_max 5 --x2_min -5 --x2_max 5
```

### Workflow with box-trained Step 2, near-zero Step 3, and max blend
```bash
lyapnn workflow --device cpu --dtype float32 \
  --step2_sample_mode box \
  --step2_box_x1_min -20 --step2_box_x1_max 20 \
  --step2_box_x2_min -20 --step2_box_x2_max 20 \
  --x1_min -5 --x1_max 5 --x2_min -5 --x2_max 5 \
  --run_step4_rect \
  --step4_outer -7 7 -7 7 \
  --step4_inner -5 5 -5 5 \
  --step4_blend_mode max
```

Outputs are written to `runs/step2/` and `runs/step3/` by default.

Notes:
- No seaborn; matplotlib only.
- `V(x) = rho(x)^mu * W(y)` with `W(y) = softplus(Wraw(y)) + eps`.

### Backward-compatible scripts
The `scripts/` folder still exists, but now it simply forwards to the unified CLI.
For open-source usage, prefer `lyapnn ...` commands.
