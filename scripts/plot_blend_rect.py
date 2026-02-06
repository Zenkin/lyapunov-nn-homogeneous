#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rectangle-based blending of step2_V and W_model.

Example:
python scripts/plot_blend_rect.py ^
  --ckpt_V runs/step2/step2_V.pt --ckpt_W runs/step3/W_model.pt ^
  --grid 401 ^
  --x1_min 0 --x1_max 10 --x2_min -10 --x2_max 2 ^
  --outer 0 10 -10 2 ^
  --inner 0 1 -1 1 ^
  --alpha 0.2 ^
  --plot_3d --save --outdir runs/step4
"""
from __future__ import annotations

import argparse

from lyapnn.viz.blend_rect import Rect
from lyapnn.pipelines.step4_blend_rect_plot import Step4Cfg, plot_step4_rect_blend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_V", required=True)
    ap.add_argument("--ckpt_W", required=True)
    ap.add_argument("--outdir", default="runs/step4")

    ap.add_argument("--x1_min", type=float, required=True)
    ap.add_argument("--x1_max", type=float, required=True)
    ap.add_argument("--x2_min", type=float, required=True)
    ap.add_argument("--x2_max", type=float, required=True)
    ap.add_argument("--grid", type=int, default=401)

    ap.add_argument("--outer", type=float, nargs=4, metavar=("x1min", "x1max", "x2min", "x2max"), required=True)
    ap.add_argument("--inner", type=float, nargs=4, metavar=("x1min", "x1max", "x2min", "x2max"), required=True)

    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--plot_3d", action="store_true")
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--no_save", action="store_true")

    args = ap.parse_args()

    outer = Rect(*args.outer)
    inner = Rect(*args.inner)

    cfg = Step4Cfg(
        ckpt_V=args.ckpt_V,
        ckpt_W=args.ckpt_W,
        outdir=args.outdir,
        x1_min=args.x1_min,
        x1_max=args.x1_max,
        x2_min=args.x2_min,
        x2_max=args.x2_max,
        grid=args.grid,
        outer=outer,
        inner=inner,
        alpha=args.alpha,
        plot_3d=bool(args.plot_3d),
        show=not bool(args.no_show),
        save=not bool(args.no_save),
    )

    info = plot_step4_rect_blend(cfg)
    print("[step4] done:", info)


if __name__ == "__main__":
    main()
