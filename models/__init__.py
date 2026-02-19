# models/__init__.py

# --- модели ---
from .nn_models import W, WArticle, build_W
from . import nn_models_cfg as cfg

# --- динамика системы ---
from .full_system_dynamics import Params, FullSystem

__all__ = [
    "W",
    "WArticle",
    "build_W",
    "Params",
    "FullSystem",
]
