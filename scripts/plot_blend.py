#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from lyapnn.cli import main

if __name__ == "__main__":
    # example:
    # python scripts/plot_blend.py step4-plot --ckpt_V runs/step2/step2_V.pt --ckpt_W runs/step3/W_model.pt \
    #   --x1_min 0 --x1_max 10 --x2_min -10 --x2_max 2 --c1 2.0 --c2 0.5 --plot_3d --save
    sys.argv = ["lyapnn"] + sys.argv[1:]
    main()
