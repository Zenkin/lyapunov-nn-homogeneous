from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Any

from lyapnn.pipelines.step2_plot import Step2PlotCfg, plot_step2
from lyapnn.training.step2_train_vinf import TrainCfg as Step2Cfg, train_step2
from lyapnn.training.step3_train_w_near_zero import TrainCfg as Step3Cfg, train_step3
from lyapnn.viz.w_diagnostics import WPlotCfg, plot_w_diagnostics_from_ckpt


@dataclass
class WorkflowCfg:
    step2_cfg: Step2Cfg
    step3_cfg: Step3Cfg
    step2_plot_cfg: Step2PlotCfg
    step3_plot_cfg: WPlotCfg
    step2_ckpt: str
    step3_ckpt: str
    run_step2_plot: bool = True
    run_step3_plot: bool = True


def run_workflow(cfg: WorkflowCfg, device: str, dtype: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(cfg.step2_ckpt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.step3_ckpt) or ".", exist_ok=True)

    train_step2(cfg.step2_cfg, save_path=cfg.step2_ckpt)
    step2_info = {}
    if cfg.run_step2_plot:
        step2_info = plot_step2(cfg.step2_plot_cfg)

    train_step3(cfg.step3_cfg, save_path=cfg.step3_ckpt)
    step3_info = {}
    if cfg.run_step3_plot:
        step3_info = plot_w_diagnostics_from_ckpt(cfg.step3_ckpt, cfg=cfg.step3_plot_cfg, device=device, dtype=dtype)

    return {"step2": step2_info, "step3": step3_info}
