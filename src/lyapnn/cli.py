from __future__ import annotations

import argparse

from lyapnn.pipelines.run import RunCfg, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lyapnn", description="End-to-end Lyapunov training/plots.")
    ap.add_argument("--outdir", default="runs/output")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_show", action="store_true", help="Do not show interactive figures.")
    ap.add_argument("--no_save", action="store_true", help="Do not save images.")
    ap.add_argument("--plot_inset", action="store_true", help="Enable zoomed inset for 3D plots.")
    inset_mode = ap.add_mutually_exclusive_group()
    inset_mode.add_argument("--inset_auto", action="store_true", help="Auto inset bbox (default).")
    inset_mode.add_argument("--inset_manual", action="store_true", help="Manual inset bbox.")
    ap.add_argument("--inset_zoom_frac", type=float, default=0.2)
    ap.add_argument("--inset_x1_min", type=float)
    ap.add_argument("--inset_x1_max", type=float)
    ap.add_argument("--inset_x2_min", type=float)
    ap.add_argument("--inset_x2_max", type=float)
    ap.add_argument("--inset_position", type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--inset_border_color", default="red")
    ap.add_argument("--inset_border_lw", type=float, default=3.0)
    ap.add_argument("--inset_connectors", action="store_true", default=True)

    ap.add_argument("--omega_x1_min", type=float, default=-20.0)
    ap.add_argument("--omega_x1_max", type=float, default=20.0)
    ap.add_argument("--omega_x2_min", type=float, default=-20.0)
    ap.add_argument("--omega_x2_max", type=float, default=20.0)

    ap.add_argument("--w_box_x1_min", type=float, default=-5.0)
    ap.add_argument("--w_box_x1_max", type=float, default=5.0)
    ap.add_argument("--w_box_x2_min", type=float, default=-5.0)
    ap.add_argument("--w_box_x2_max", type=float, default=5.0)

    ap.add_argument("--x_box_x1_min", type=float, default=-1.0)
    ap.add_argument("--x_box_x1_max", type=float, default=1.0)
    ap.add_argument("--x_box_x2_min", type=float, default=-1.0)
    ap.add_argument("--x_box_x2_max", type=float, default=1.0)

    ap.add_argument("--grid", type=int, default=301)

    ap.add_argument("--vinf_mu", type=float, default=2.0)
    ap.add_argument("--vinf_alpha", type=float, default=1.0)
    ap.add_argument("--vinf_hidden", type=int, default=64)
    ap.add_argument("--vinf_depth", type=int, default=3)
    ap.add_argument("--vinf_steps", type=int, default=1000)
    ap.add_argument("--vinf_batch", type=int, default=2048)
    ap.add_argument("--vinf_lr", type=float, default=2e-4)
    ap.add_argument("--vinf_log_every", type=int, default=200)
    ap.add_argument("--vinf_normalize_margin", type=int, default=1, help="1/0")

    ap.add_argument("--w_hidden", type=int, default=64)
    ap.add_argument("--w_depth", type=int, default=2)
    ap.add_argument("--w_steps", type=int, default=50000)
    ap.add_argument("--w_batch", type=int, default=2048)
    ap.add_argument("--w_lr", type=float, default=1e-3)
    ap.add_argument("--w_log_every", type=int, default=200)
    ap.add_argument("--w_r_min", type=float, default=0.0)
    ap.add_argument("--w_margin", type=float, default=0.0)
    ap.add_argument("--w_alpha_pos", type=float, default=1e-3)
    ap.add_argument("--w_eps_s", type=float, default=1e-2)
    ap.add_argument("--w_lam_s", type=float, default=1e-3)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    cfg = RunCfg(
        outdir=args.outdir,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        omega=(args.omega_x1_min, args.omega_x1_max, args.omega_x2_min, args.omega_x2_max),
        w_box=(args.w_box_x1_min, args.w_box_x1_max, args.w_box_x2_min, args.w_box_x2_max),
        x_box=(args.x_box_x1_min, args.x_box_x1_max, args.x_box_x2_min, args.x_box_x2_max),
        grid=args.grid,
        vinf_mu=args.vinf_mu,
        vinf_alpha=args.vinf_alpha,
        vinf_hidden=args.vinf_hidden,
        vinf_depth=args.vinf_depth,
        vinf_steps=args.vinf_steps,
        vinf_batch=args.vinf_batch,
        vinf_lr=args.vinf_lr,
        vinf_log_every=args.vinf_log_every,
        vinf_normalize_margin=bool(args.vinf_normalize_margin),
        w_hidden=args.w_hidden,
        w_depth=args.w_depth,
        w_steps=args.w_steps,
        w_batch=args.w_batch,
        w_lr=args.w_lr,
        w_log_every=args.w_log_every,
        w_r_min=args.w_r_min,
        w_margin=args.w_margin,
        w_alpha_pos=args.w_alpha_pos,
        w_eps_s=args.w_eps_s,
        w_lam_s=args.w_lam_s,
        plot_inset=args.plot_inset,
        inset_auto=args.inset_auto or not args.inset_manual,
        inset_zoom_frac=args.inset_zoom_frac,
        inset_manual=args.inset_manual,
        inset_x1_min=args.inset_x1_min,
        inset_x1_max=args.inset_x1_max,
        inset_x2_min=args.inset_x2_min,
        inset_x2_max=args.inset_x2_max,
        inset_position=tuple(args.inset_position) if args.inset_position else (0.58, 0.62, 0.33, 0.3),
        inset_border_color=args.inset_border_color,
        inset_border_lw=args.inset_border_lw,
        inset_connectors=args.inset_connectors,
        show=not args.no_show,
        save=not args.no_save,
    )
    run_pipeline(cfg)
