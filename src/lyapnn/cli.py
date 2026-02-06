from __future__ import annotations

import argparse
import os

from lyapnn.training.step2_train_vinf import TrainCfg as Step2Cfg, train_step2
from lyapnn.training.step3_train_w_near_zero import TrainCfg as Step3Cfg, train_step3
from lyapnn.pipelines.step2_plot import Step2PlotCfg, plot_step2
from lyapnn.viz.w_diagnostics import WPlotCfg, plot_w_diagnostics_from_ckpt
from lyapnn.pipelines.step4_blend_plot import Step4PlotCfg, plot_step4_blend
from lyapnn.viz.blend_rect import Rect
from lyapnn.pipelines.step4_blend_rect_plot import Step4Cfg, plot_step4_rect_blend
from lyapnn.pipelines.workflow import WorkflowCfg, run_workflow


def _cmd_step2_train(a: argparse.Namespace) -> None:
    os.makedirs(a.outdir, exist_ok=True)
    ckpt = a.ckpt or os.path.join(a.outdir, "step2_V.pt")
    cfg = Step2Cfg(
        seed=a.seed,
        device=a.device,
        mu=a.mu,
        alpha=a.alpha,
        hidden=a.hidden,
        depth=a.depth,
        steps=a.steps,
        batch=a.batch,
        lr=a.lr,
        log_every=a.log_every,
        normalize_margin=bool(a.normalize_margin),
        sample_mode=a.sample_mode,
        box_x1_min=a.box_x1_min,
        box_x1_max=a.box_x1_max,
        box_x2_min=a.box_x2_min,
        box_x2_max=a.box_x2_max,
    )
    train_step2(cfg, save_path=ckpt)


def _cmd_step2_plot(a: argparse.Namespace) -> None:
    any_sel = a.inf or a.full or a.bad_regions or a.plot_3d
    cfg = Step2PlotCfg(
        ckpt=a.ckpt,
        device=a.device,
        outdir=a.outdir,
        plot_inf=(a.inf or not any_sel),
        plot_full=(a.full or not any_sel),
        plot_bad_regions=(a.bad_regions or not any_sel),
        plot_3d=(a.plot_3d or not any_sel),
        n=a.n,
        alpha_for_margin=a.alpha_for_margin,
    )
    info = plot_step2(cfg)
    print(f"[done] outdir={info['outdir']} x_eq={info['x_eq']:.12g}")


def _cmd_step3_train(a: argparse.Namespace) -> None:
    os.makedirs(a.outdir, exist_ok=True)
    ckpt = a.ckpt or os.path.join(a.outdir, "W_model.pt")
    cfg = Step3Cfg(
        seed=a.seed,
        device=a.device,
        dtype=a.dtype,
        hidden=a.hidden,
        depth=a.depth,
        steps=a.steps,
        batch=a.batch,
        lr=a.lr,
        log_every=a.log_every,
        x1_min=a.x1_min,
        x1_max=a.x1_max,
        x2_min=a.x2_min,
        x2_max=a.x2_max,
        r_min=a.r_min,
        margin=a.margin,
        alpha_pos=a.alpha_pos,
        eps_s=a.eps_s,
        lam_s=a.lam_s,
    )
    train_step3(cfg, save_path=ckpt)


def _cmd_step3_plot(a: argparse.Namespace) -> None:
    cfg = WPlotCfg(
        x1_min=a.x1_min, x1_max=a.x1_max,
        x2_min=a.x2_min, x2_max=a.x2_max,
        grid=a.grid,
        alpha=a.alpha,
        plot_3d=bool(a.plot_3d),
        save_path_4=(os.path.join(a.outdir, "W_diag4.png") if a.save else None),
        save_path_bad=(os.path.join(a.outdir, "W_bad_regions.png") if a.save else None),
        save_path_3d_w=(os.path.join(a.outdir, "W_3d.png") if a.save else None),
        save_path_3d_wdot=(os.path.join(a.outdir, "Wdot_3d.png") if a.save else None),
    )
    info = plot_w_diagnostics_from_ckpt(a.ckpt, cfg=cfg, device=a.device, dtype=a.dtype)
    print(f"[done] x_eq={info.get('x_eq')}")


def _cmd_workflow(a: argparse.Namespace) -> None:
    step2_outdir = a.step2_outdir
    step3_outdir = a.step3_outdir
    os.makedirs(step2_outdir, exist_ok=True)
    os.makedirs(step3_outdir, exist_ok=True)

    step2_ckpt = os.path.join(step2_outdir, "step2_V.pt")
    step3_ckpt = os.path.join(step3_outdir, "W_model.pt")

    step2_cfg = Step2Cfg(
        seed=a.seed,
        device=a.device,
        mu=a.step2_mu,
        alpha=a.step2_alpha,
        hidden=a.step2_hidden,
        depth=a.step2_depth,
        steps=a.step2_steps,
        batch=a.step2_batch,
        lr=a.step2_lr,
        log_every=a.step2_log_every,
        normalize_margin=bool(a.step2_normalize_margin),
        sample_mode=a.step2_sample_mode,
        box_x1_min=a.step2_box_x1_min,
        box_x1_max=a.step2_box_x1_max,
        box_x2_min=a.step2_box_x2_min,
        box_x2_max=a.step2_box_x2_max,
    )

    step2_plot_cfg = Step2PlotCfg(
        ckpt=step2_ckpt,
        device=a.device,
        outdir=step2_outdir,
        plot_inf=not a.no_step2_plot,
        plot_full=not a.no_step2_plot,
        plot_bad_regions=not a.no_step2_plot,
        plot_3d=bool(a.step2_plot_3d),
        n=a.step2_plot_n,
        alpha_for_margin=a.step2_plot_alpha,
        inf_x1_lim=(a.step2_inf_x1_min, a.step2_inf_x1_max),
        inf_x2_lim=(a.step2_inf_x2_min, a.step2_inf_x2_max),
        full_x1_lim=(a.step2_full_x1_min, a.step2_full_x1_max),
        full_x2_lim=(a.step2_full_x2_min, a.step2_full_x2_max),
    )

    step3_cfg = Step3Cfg(
        seed=a.seed,
        device=a.device,
        dtype=a.dtype,
        hidden=a.step3_hidden,
        depth=a.step3_depth,
        steps=a.step3_steps,
        batch=a.step3_batch,
        lr=a.step3_lr,
        log_every=a.step3_log_every,
        x1_min=a.x1_min,
        x1_max=a.x1_max,
        x2_min=a.x2_min,
        x2_max=a.x2_max,
        r_min=a.step3_r_min,
        margin=a.step3_margin,
        alpha_pos=a.step3_alpha_pos,
        eps_s=a.step3_eps_s,
        lam_s=a.step3_lam_s,
    )

    step3_plot_cfg = WPlotCfg(
        x1_min=a.x1_min,
        x1_max=a.x1_max,
        x2_min=a.x2_min,
        x2_max=a.x2_max,
        grid=a.step3_plot_grid,
        alpha=a.step3_plot_alpha,
        plot_3d=bool(a.step3_plot_3d),
        save_path_4=os.path.join(step3_outdir, "W_diag4.png"),
        save_path_bad=os.path.join(step3_outdir, "W_bad_regions.png"),
        save_path_3d_w=os.path.join(step3_outdir, "W_3d.png"),
        save_path_3d_wdot=os.path.join(step3_outdir, "Wdot_3d.png"),
    )

    cfg = WorkflowCfg(
        step2_cfg=step2_cfg,
        step3_cfg=step3_cfg,
        step2_plot_cfg=step2_plot_cfg,
        step3_plot_cfg=step3_plot_cfg,
        step2_ckpt=step2_ckpt,
        step3_ckpt=step3_ckpt,
        run_step2_plot=not a.no_step2_plot,
        run_step3_plot=not a.no_step3_plot,
        run_step4_rect=bool(a.run_step4_rect),
        step4_cfg=(
            Step4Cfg(
                ckpt_V=step2_ckpt,
                ckpt_W=step3_ckpt,
                outdir=a.step4_outdir,
                x1_min=a.step4_x1_min,
                x1_max=a.step4_x1_max,
                x2_min=a.step4_x2_min,
                x2_max=a.step4_x2_max,
                grid=a.step4_grid,
                outer=Rect(*a.step4_outer),
                inner=Rect(*a.step4_inner),
                alpha=a.step4_alpha,
                plot_3d=bool(a.step4_plot_3d),
                show=not bool(a.step4_no_show),
                save=not bool(a.step4_no_save),
                blend_mode=a.step4_blend_mode,
                w_scale=a.step4_w_scale,
            )
            if a.run_step4_rect
            else None
        ),
    )
    run_workflow(cfg, device=a.device, dtype=a.dtype)
    print(f"[workflow] step2_outdir={step2_outdir} step3_outdir={step3_outdir}")


def _cmd_step4_plot(a: argparse.Namespace) -> None:
    os.makedirs(a.outdir, exist_ok=True)
    cfg = Step4PlotCfg(
        ckpt_V=a.ckpt_V,
        ckpt_W=a.ckpt_W,
        outdir=a.outdir,
        device=a.device,
        dtype=a.dtype,
        n=a.n,
        x1_lim=(a.x1_lim_min, a.x1_lim_max),
        x2_lim=(a.x2_lim_min, a.x2_lim_max),
        x1_min=a.x1_min,
        x1_max=a.x1_max,
        x2_min=a.x2_min,
        x2_max=a.x2_max,
        c1=a.c1,
        c2=a.c2,
        band_rel=a.band_rel,
        alpha=a.alpha,
        plot_3d=bool(a.plot_3d),
        save=bool(a.save),
    )
    info = plot_step4_blend(cfg)
    print(f"[done] outdir={info['outdir']} x_eq={info['x_eq']:.12g} k={info['k']:.6g}")


def _cmd_step4_rect_plot(args) -> None:
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
        blend_mode=args.blend_mode,
        w_scale=args.w_scale,
    )

    info = plot_step4_rect_blend(cfg)
    print("[step4-rect] done:", info)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lyapnn", description="NN-based Lyapunov diagnostics (Step2/Step3).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- Step 2 ----
    p2t = sub.add_parser("step2-train", help="Train homogeneous V on f_inf over S_r(1).")
    p2t.add_argument("--outdir", default="runs/step2")
    p2t.add_argument("--ckpt", default=None, help="Override checkpoint path (default: outdir/step2_V.pt)")
    p2t.add_argument("--device", default="cpu")
    p2t.add_argument("--seed", type=int, default=0)
    p2t.add_argument("--mu", type=float, default=2.0)
    p2t.add_argument("--alpha", type=float, default=1.0)
    p2t.add_argument("--hidden", type=int, default=64)
    p2t.add_argument("--depth", type=int, default=3)
    p2t.add_argument("--steps", type=int, default=1000)
    p2t.add_argument("--batch", type=int, default=2048)
    p2t.add_argument("--lr", type=float, default=2e-4)
    p2t.add_argument("--log_every", type=int, default=200)
    p2t.add_argument("--normalize_margin", type=int, default=1, help="1/0")
    p2t.add_argument("--sample_mode", default="sr1", choices=["sr1", "box"])
    p2t.add_argument("--box_x1_min", type=float, default=-20.0)
    p2t.add_argument("--box_x1_max", type=float, default=20.0)
    p2t.add_argument("--box_x2_min", type=float, default=-20.0)
    p2t.add_argument("--box_x2_max", type=float, default=20.0)
    p2t.set_defaults(func=_cmd_step2_train)

    p2p = sub.add_parser("step2-plot", help="Plot diagnostics for f_inf and full system from a V checkpoint.")
    p2p.add_argument("--ckpt", required=True)
    p2p.add_argument("--outdir", default="runs/step2")
    p2p.add_argument("--device", default="cpu")
    p2p.add_argument("--n", type=int, default=301)
    p2p.add_argument("--alpha_for_margin", type=float, default=0.2)
    p2p.add_argument("--inf", action="store_true", help="Only/also plot f_inf diagnostics")
    p2p.add_argument("--full", action="store_true", help="Only/also plot full-system diag4")
    p2p.add_argument("--bad_regions", action="store_true", help="Only/also plot hatched Vdot>=0 map")
    p2p.add_argument("--plot_3d", action="store_true", help="Only/also plot interactive 3D surfaces")
    p2p.set_defaults(func=_cmd_step2_plot)

    # ---- Step 3 ----
    p3t = sub.add_parser("step3-train", help="Train near-zero W(x)=||T(x)||^2 on a shifted rectangle.")
    p3t.add_argument("--outdir", default="runs/step3")
    p3t.add_argument("--ckpt", default=None, help="Override checkpoint path (default: outdir/W_model.pt)")
    p3t.add_argument("--device", default="cpu")
    p3t.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p3t.add_argument("--seed", type=int, default=0)
    p3t.add_argument("--hidden", type=int, default=64)
    p3t.add_argument("--depth", type=int, default=2)
    p3t.add_argument("--steps", type=int, default=50000)
    p3t.add_argument("--batch", type=int, default=2048)
    p3t.add_argument("--lr", type=float, default=1e-3)
    p3t.add_argument("--log_every", type=int, default=200)
    p3t.add_argument("--x1_min", type=float, default=-5.0)
    p3t.add_argument("--x1_max", type=float, default=5.0)
    p3t.add_argument("--x2_min", type=float, default=-5.0)
    p3t.add_argument("--x2_max", type=float, default=5.0)
    p3t.add_argument("--r_min", type=float, default=0.0)
    p3t.add_argument("--margin", type=float, default=0.0)
    p3t.add_argument("--alpha_pos", type=float, default=1e-3)
    p3t.add_argument("--eps_s", type=float, default=1e-2)
    p3t.add_argument("--lam_s", type=float, default=1e-3)
    p3t.set_defaults(func=_cmd_step3_train)

    p3p = sub.add_parser("step3-plot", help="Plot W diagnostics from a saved checkpoint.")
    p3p.add_argument("--ckpt", required=True)
    p3p.add_argument("--outdir", default="runs/step3")
    p3p.add_argument("--device", default="cpu")
    p3p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p3p.add_argument("--x1_min", type=float, default=-5.0)
    p3p.add_argument("--x1_max", type=float, default=5.0)
    p3p.add_argument("--x2_min", type=float, default=-5.0)
    p3p.add_argument("--x2_max", type=float, default=5.0)
    p3p.add_argument("--grid", type=int, default=401)
    p3p.add_argument("--alpha", type=float, default=1.0)
    p3p.add_argument("--plot_3d", action="store_true")
    p3p.add_argument("--save", action="store_true", help="Save figures to outdir")
    p3p.set_defaults(func=_cmd_step3_plot)

    # ---- Workflow ----
    wf = sub.add_parser("workflow", help="Run step2-train+plot and step3-train+plot with unified settings.")
    wf.add_argument("--step2_outdir", default="runs/step2")
    wf.add_argument("--step3_outdir", default="runs/step3")
    wf.add_argument("--device", default="cpu")
    wf.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    wf.add_argument("--seed", type=int, default=0)

    wf.add_argument("--step2_mu", type=float, default=2.0)
    wf.add_argument("--step2_alpha", type=float, default=1.0)
    wf.add_argument("--step2_hidden", type=int, default=64)
    wf.add_argument("--step2_depth", type=int, default=3)
    wf.add_argument("--step2_steps", type=int, default=1000)
    wf.add_argument("--step2_batch", type=int, default=2048)
    wf.add_argument("--step2_lr", type=float, default=2e-4)
    wf.add_argument("--step2_log_every", type=int, default=200)
    wf.add_argument("--step2_normalize_margin", type=int, default=1, help="1/0")
    wf.add_argument("--step2_plot_n", type=int, default=301)
    wf.add_argument("--step2_plot_alpha", type=float, default=0.2)
    wf.add_argument("--step2_plot_3d", action="store_true")
    wf.add_argument("--no_step2_plot", action="store_true")
    wf.add_argument("--step2_sample_mode", default="box", choices=["sr1", "box"])
    wf.add_argument("--step2_box_x1_min", type=float, default=-20.0)
    wf.add_argument("--step2_box_x1_max", type=float, default=20.0)
    wf.add_argument("--step2_box_x2_min", type=float, default=-20.0)
    wf.add_argument("--step2_box_x2_max", type=float, default=20.0)
    wf.add_argument("--step2_inf_x1_min", type=float, default=-6.0)
    wf.add_argument("--step2_inf_x1_max", type=float, default=6.0)
    wf.add_argument("--step2_inf_x2_min", type=float, default=-6.0)
    wf.add_argument("--step2_inf_x2_max", type=float, default=6.0)
    wf.add_argument("--step2_full_x1_min", type=float, default=-20.0)
    wf.add_argument("--step2_full_x1_max", type=float, default=20.0)
    wf.add_argument("--step2_full_x2_min", type=float, default=-20.0)
    wf.add_argument("--step2_full_x2_max", type=float, default=20.0)

    wf.add_argument("--step3_hidden", type=int, default=64)
    wf.add_argument("--step3_depth", type=int, default=2)
    wf.add_argument("--step3_steps", type=int, default=50000)
    wf.add_argument("--step3_batch", type=int, default=2048)
    wf.add_argument("--step3_lr", type=float, default=1e-3)
    wf.add_argument("--step3_log_every", type=int, default=200)
    wf.add_argument("--step3_r_min", type=float, default=0.0)
    wf.add_argument("--step3_margin", type=float, default=0.0)
    wf.add_argument("--step3_alpha_pos", type=float, default=1e-3)
    wf.add_argument("--step3_eps_s", type=float, default=1e-2)
    wf.add_argument("--step3_lam_s", type=float, default=1e-3)

    wf.add_argument("--x1_min", type=float, default=-5.0)
    wf.add_argument("--x1_max", type=float, default=5.0)
    wf.add_argument("--x2_min", type=float, default=-5.0)
    wf.add_argument("--x2_max", type=float, default=5.0)

    wf.add_argument("--step3_plot_grid", type=int, default=401)
    wf.add_argument("--step3_plot_alpha", type=float, default=1.0)
    wf.add_argument("--step3_plot_3d", action="store_true")
    wf.add_argument("--no_step3_plot", action="store_true")
    wf.add_argument("--run_step4_rect", action="store_true", help="Run step4-rect blend after step3.")
    wf.add_argument("--step4_outdir", default="runs/step4")
    wf.add_argument("--step4_x1_min", type=float, default=-10.0)
    wf.add_argument("--step4_x1_max", type=float, default=10.0)
    wf.add_argument("--step4_x2_min", type=float, default=-10.0)
    wf.add_argument("--step4_x2_max", type=float, default=10.0)
    wf.add_argument("--step4_grid", type=int, default=401)
    wf.add_argument("--step4_outer", type=float, nargs=4, default=[-7.0, 7.0, -7.0, 7.0])
    wf.add_argument("--step4_inner", type=float, nargs=4, default=[-5.0, 5.0, -5.0, 5.0])
    wf.add_argument("--step4_alpha", type=float, default=0.2)
    wf.add_argument("--step4_plot_3d", action="store_true")
    wf.add_argument("--step4_no_show", action="store_true")
    wf.add_argument("--step4_no_save", action="store_true")
    wf.add_argument("--step4_blend_mode", default="max", choices=["smooth", "max"])
    wf.add_argument("--step4_w_scale", type=float, default=1.0)
    wf.set_defaults(func=_cmd_workflow)



    # ---- Step 4 ----
    p4p = sub.add_parser("step4-plot", help="Blend Step2 V_inf with Step3 W near zero and plot diagnostics.")
    p4p.add_argument("--ckpt_V", required=True, help="Path to Step2 V checkpoint (step2_V.pt)")
    p4p.add_argument("--ckpt_W", required=True, help="Path to Step3 W checkpoint (W_model.pt)")
    p4p.add_argument("--outdir", default="runs/step4")
    p4p.add_argument("--device", default="cpu")
    p4p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p4p.add_argument("--n", type=int, default=301)
    p4p.add_argument("--x1_lim_min", type=float, default=-20.0)
    p4p.add_argument("--x1_lim_max", type=float, default=20.0)
    p4p.add_argument("--x2_lim_min", type=float, default=-20.0)
    p4p.add_argument("--x2_lim_max", type=float, default=20.0)

    # rectangle where blending/W is allowed (SHIFTED coords)
    p4p.add_argument("--x1_min", type=float, required=True)
    p4p.add_argument("--x1_max", type=float, required=True)
    p4p.add_argument("--x2_min", type=float, required=True)
    p4p.add_argument("--x2_max", type=float, required=True)

    # V levels for blending (based on V_inf values)
    p4p.add_argument("--c1", type=float, required=True, help="Start blending at V_inf=c1 (outside uses V_inf)")
    p4p.add_argument("--c2", type=float, required=True, help="Finish blending at V_inf=c2 (inside uses W)")
    p4p.add_argument("--band_rel", type=float, default=0.03, help="Relative band around c1 to pick real scale points")

    p4p.add_argument("--alpha", type=float, default=0.2, help="margin = Vdot + alpha*V")
    p4p.add_argument("--plot_3d", action="store_true")
    p4p.add_argument("--save", action="store_true")
    p4p.set_defaults(func=_cmd_step4_plot)

    p4 = sub.add_parser(
        "step4-rect-plot",
        help="Blend step2_V and W_model using two nested rectangles in SHIFTED coords."
    )

    p4.add_argument("--ckpt_V", required=True, help="Path to step2 V checkpoint (e.g. runs/step2/step2_V.pt)")
    p4.add_argument("--ckpt_W", required=True, help="Path to step3 W checkpoint (e.g. runs/step3/W_model.pt)")
    p4.add_argument("--outdir", default="runs/step4", help="Output directory")

    p4.add_argument("--x1_min", type=float, required=True)
    p4.add_argument("--x1_max", type=float, required=True)
    p4.add_argument("--x2_min", type=float, required=True)
    p4.add_argument("--x2_max", type=float, required=True)
    p4.add_argument("--grid", type=int, default=401)

    p4.add_argument("--outer", type=float, nargs=4, required=True, metavar=("x1min", "x1max", "x2min", "x2max"),
                    help="Outer rectangle in SHIFTED coords: outside -> V only")
    p4.add_argument("--inner", type=float, nargs=4, required=True, metavar=("x1min", "x1max", "x2min", "x2max"),
                    help="Inner rectangle in SHIFTED coords: inside -> W only")

    p4.add_argument("--alpha", type=float, default=0.2, help="Margin alpha for: Vdot + alpha*V")
    p4.add_argument("--blend_mode", default="smooth", choices=["smooth", "max"])
    p4.add_argument("--w_scale", type=float, default=1.0, help="Scale factor for W in max blend mode")
    p4.add_argument("--plot_3d", action="store_true", help="Also plot 3D surfaces")
    p4.add_argument("--no_show", action="store_true", help="Do not show plots interactively")
    p4.add_argument("--no_save", action="store_true", help="Do not save images")

    p4.set_defaults(func=_cmd_step4_rect_plot)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)
