"""Constans for running the scripts.
"""

import os
import pathlib

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

ASSETS_FOLDER = pathlib.Path(__file__).parent / "assets"
