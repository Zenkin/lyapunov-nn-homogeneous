# models/__init__.py

# --- модели ---
from .nn_models import WArticle, build_W
from . import nn_models_cfg as cfg
from .losses import loss_L2_article

# --- динамика системы ---
from .full_system_dynamics import Params, FullSystem

__all__ = [
    "WArticle",
    "build_W",
    "Params",
    "FullSystem",
    "loss_L2_article",
]
