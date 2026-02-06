#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible wrapper.

Prefer:
    lyapnn step3-train --x1_min ... --x1_max ... --x2_min ... --x2_max ...
    lyapnn step3-plot  --ckpt runs/step3/W_model.pt --x1_min ... --x1_max ... --x2_min ... --x2_max ...
"""

from lyapnn.cli import main


if __name__ == "__main__":
    main()
