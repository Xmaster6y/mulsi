"""Constans for running the scripts."""

import os
import pathlib

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

ASSETS_FOLDER = pathlib.Path(__file__).parent / "assets"

SPLITS = ["train", "test"]
DATASET_NAME = "mulsi/fruit-vegetable-concepts"
CONCEPTS = [
    # Environment
    "stem",
    "leaf",
    "tail",
    "seed",
    "pulp",
    "soil",
    "tree",
    # Shapes
    "ovaloid",
    "sphere",
    "cylinder",
    "cube",
    # Colors
    "black",
    "purple",
    "red",
    "blue",
    "green",
    "brown",
    "orange",
    "yellow",
    "white",
]
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
    "ginger": ["brown", "cylinder"],
    "jalepeno": ["stem", "green", "cylinder"],
    "mango": ["yellow", "ovaloid"],
    "orange": ["orange", "sphere"],
    "eggplant": ["purple", "cylinder"],
    "cauliflower": ["white", "leaf"],
    "tomato": [],  # already labeled
    "kiwi": [],  # already labeled
    "peas": ["seed", "green", "sphere"],
    "potato": ["brown", "ovaloid"],
    "lemon": [],  # already labeled
    "chilli pepper": ["red", "stem", "cylinder"],
    "watermelon": [],  # already labeled
    "apple": ["red", "green", "sphere"],
    "lettuce": [],  # already labeled
    "banana": ["yellow", "cylinder"],
    "corn": ["seed", "yellow", "cylinder"],
    "cabbage": [],  # already labeled
    "capsicum": ["green", "cylinder"],
    "spinach": ["leaf", "green"],
    "garlic": ["white", "ovaloid"],
    "soy beans": ["seed", "brown"],
    "grapes": ["green", "sphere"],
    "carrot": ["orange", "stem", "cylinder"],
    "paprika": [],  # already labeled
    "beetroot": [],  # already labeled
    "turnip": ["white", "purple", "stem", "sphere", "tail"],
    "pineapple": ["ovaloid", "brown", "yellow"],
    "bell pepper": ["green", "red", "yellow", "stem", "cylinder"],
    "raddish": ["red", "stem", "sphere"],
    "onion": ["white", "sphere"],
    "pear": ["green", "ovaloid"],
    "pomegranate": ["red", "seed", "sphere"],
    "sweetcorn": ["seed", "yellow", "cylinder"],
    "sweetpotato": ["orange", "brown", "ovaloid"],
}

LABELED_CLASSES = [
    "tomato",
    "kiwi",
    "lemon",
    "watermelon",
    "paprika",
    "lettuce",
    "beetroot",
    "cabbage",
    "bell pepper",
    "carrot",
]

assert len(CLASSES) == len(CLASS_CONCEPTS_VALUES.keys())
