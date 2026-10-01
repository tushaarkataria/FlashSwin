# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# FlashSwin release: the research version of this file also imported the SimMIM pre-training
# loaders. SimMIM is not part of this release, so importing them here would make `from data
# import build_loader` -- and therefore main.py -- fail at startup with ModuleNotFoundError.
from .build import build_loader
