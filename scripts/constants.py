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
    "cucumber",
    "ginger",
    "jalepeno",
    "mango",
    "orange",
    "eggplant",
    "cauliflower",
    "tomato",
    "kiwi",
    "peas",
    "potato",
    "lemon",
    "chilli pepper",
    "watermelon",
    "apple",
    "lettuce",
    "banana",
    "corn",
    "cabbage",
    "capsicum",
    "spinach",
    "garlic",
    "soy beans",
    "grapes",
    "carrot",
    "paprika",
    "beetroot",
    "turnip",
    "pineapple",
    "bell pepper",
    "raddish",
    "onion",
    "pear",
    "pomegranate",
    "sweetcorn",
    "sweetpotato",
]

CLASS_CONCEPTS_VALUES = {
    "cucumber": ["green", "cylinder"],
    "ginger": ["brown"],
    "jalepeno": ["stem", "green", "cylinder"],
    "mango": ["seed", "orange", "ovaloid"],
    "orange": ["orange", "sphere"],
    "eggplant": ["purple", "cylinder"],
    "cauliflower": ["white"],
    "tomato": [],  # already labeled
    "kiwi": [],  # already labeled
    "peas": ["seed", "green", "sphere"],
    "potato": ["brown", "ovaloid"],
    "lemon": [],  # already labeled
    "chilli pepper": ["red"],
    "watermelon": [],  # already labeled
    "apple": ["red", "green"],
    "lettuce": ["leaf", "green"],
    "banana": ["yellow", "cylinder"],
    "corn": ["seed", "yellow"],
    "cabbage": ["green"],
    "capsicum": ["green"],
    "spinach": ["leaf", "green"],
    "garlic": ["white"],
    "soy beans": ["seed", "brown"],
    "grapes": ["green"],
    "carrot": ["orange", "stem"],
    "paprika": ["red", "stem"],
    "beetroot": ["red", "stem", "tail"],
    "turnip": ["white", "purple", "stem"],
    "pineapple": ["ovaloid", "brown", "yellow"],
    "bell pepper": ["green", "red", "yellow", "stem"],
    "raddish": ["red", "stem"],
    "onion": ["white", "sphere"],
    "pear": ["green"],
    "pomegranate": ["red", "seed"],
    "sweetcorn": ["seed", "yellow"],
    "sweetpotato": ["orange", "brown", "ovaloid"],
}

assert len(CLASSES) == len(CLASS_CONCEPTS_VALUES.keys())
