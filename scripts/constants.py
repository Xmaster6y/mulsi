"""Constans for running the scripts.
"""

import os
import pathlib

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

ASSETS_FOLDER = pathlib.Path(__file__).parent / "assets"

SPLITS = ["train", "test"]
DATASET_NAME = "mulsi/fruit-vegetable-concepts"
CLASSES = [
    'cucumber', 'ginger', 'jalepeno', 'mango', 'orange', 'eggplant',
    'cauliflower', 'tomato', 'kiwi', 'peas', 'potato', 'lemon',
    'chilli pepper', 'watermelon', 'apple', 'lettuce', 'banana',
    'corn', 'cabbage', 'capsicum', 'spinach', 'garlic', 'soy beans',
    'grapes', 'carrot', 'paprika', 'beetroot', 'turnip', 'pineapple',
    'bell pepper', 'raddish', 'onion', 'pear', 'pomegranate',
    'sweetcorn', 'sweetpotato'
]