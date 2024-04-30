"""Script for pre-labeling the dataset.
"""

import argparse

import jsonlines
import pandas as pd
from huggingface_hub import HfApi
from loguru import logger

from scripts.constants import (
    ASSETS_FOLDER,
    CLASSES,
    DATASET_NAME,
    HF_TOKEN,
    SPLITS,
)

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


def get_metadata(hf_api: HfApi, split: str):
    metadata = []
    hf_api.hf_hub_download(
        repo_id=DATASET_NAME,
        filename="metadata.jsonl",
        subfolder=f"data/{split}",
        repo_type="dataset",
        local_dir=f"{ASSETS_FOLDER}/{DATASET_NAME}",
    )

    with jsonlines.open(
        f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/metadata.jsonl"
    ) as reader:
        for row in reader:
            metadata.append(row)
    return metadata


def save_metadata(
    hf_api: HfApi, metadata: dict, split: str, push_to_hub: bool = False
):
    with jsonlines.open(
        f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/metadata.jsonl", mode="w"
    ) as writer:
        writer.write_all(metadata)

    if push_to_hub:
        hf_api.upload_file(
            path_or_fileobj=f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/metadata.jsonl",
            path_in_repo=f"data/{split}/metadata.jsonl",
            repo_id=DATASET_NAME,
            repo_type="dataset",
        )


def main(args):
    hf_api = HfApi(token=HF_TOKEN)
    for split in SPLITS:
        logger.info("Get metadata from Hub")
        metadata = get_metadata(hf_api, split=split)

        logger.info("Pre-label concepts")
        df = pd.DataFrame.from_records(metadata)

        assert len(df["class"].unique()) == len(CLASS_CONCEPTS_VALUES.keys())

        for key, values in CLASS_CONCEPTS_VALUES.items():
            for val in values:
                df.loc[df["class"] == key, val] = True
        metadata = df.to_dict(orient="records")

        logger.info("Save metadata to Hub/Locally")
        save_metadata(hf_api, metadata, split, args.push_to_hub)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("pre-label-dataset")
    parser.add_argument(
        "--push_to_hub", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
