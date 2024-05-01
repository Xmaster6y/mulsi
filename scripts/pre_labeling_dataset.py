"""Script for pre-labeling the dataset."""

import os
import json
import argparse

import jsonlines
import pandas as pd
from huggingface_hub import HfApi
from loguru import logger

from scripts.constants import (
    ASSETS_FOLDER,
    CLASS_CONCEPTS_VALUES,
    DATASET_NAME,
    HF_TOKEN,
    SPLITS,
    CONCEPTS,
    USERS,
)


def get_metadata(hf_api: HfApi, split: str):
    metadata = []
    hf_api.hf_hub_download(
        repo_id=DATASET_NAME,
        filename="metadata.jsonl",
        subfolder=f"data/{split}",
        repo_type="dataset",
        local_dir=f"{ASSETS_FOLDER}/{DATASET_NAME}",
    )

    with jsonlines.open(f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/metadata.jsonl") as reader:
        for row in reader:
            metadata.append(row)
    return metadata


def save_metadata(hf_api: HfApi, metadata: dict, split: str, push_to_hub: bool = False):
    with jsonlines.open(f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/metadata.jsonl", mode="w") as writer:
        writer.write_all(metadata)

    if push_to_hub:
        hf_api.upload_file(
            path_or_fileobj=f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/" "metadata.jsonl",
            path_in_repo=f"data/{split}/metadata.jsonl",
            repo_id=DATASET_NAME,
            repo_type="dataset",
        )


def get_votes(hf_api: HfApi):
    hf_api.snapshot_download(
        local_dir=f"{ASSETS_FOLDER}/{DATASET_NAME}",
        repo_id=DATASET_NAME,
        repo_type="dataset",
    )
    metadata = {}
    for split in SPLITS:
        metadata[split] = []
        with jsonlines.open(f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/metadata.jsonl") as reader:
            for row in reader:
                metadata[split].append(row)
    votes = {}
    for filename in os.listdir(f"{ASSETS_FOLDER}/{DATASET_NAME}/votes"):
        with open(f"{ASSETS_FOLDER}/{DATASET_NAME}/votes/{filename}") as f:
            key = filename.split(".")[0]
            votes[key] = json.load(f)
    return metadata, votes


def get_pre_labeled_concepts(item: dict):
    active_concepts = CLASS_CONCEPTS_VALUES[item["class"]]
    return {c: c in active_concepts for c in CONCEPTS}


def main(args):
    hf_api = HfApi(token=HF_TOKEN)

    logger.info("Download metadata and votes")
    metadata, votes = get_votes(hf_api)

    for split in SPLITS:
        for item in metadata[split]:
            key = item["id"]
            if key not in votes.keys():
                concepts = get_pre_labeled_concepts(item)
                votes[key] = {user: concepts for user in USERS}

    logger.info("Save votes locally")
    for key in votes:
        with open(f"{ASSETS_FOLDER}/{DATASET_NAME}/votes/{key}.json", "w") as f:
            json.dump(votes[key], f)

    if args.push_to_hub:
        logger.info("Upload votes to Hub")
        hf_api.upload_folder(
            folder_path=f"{ASSETS_FOLDER}/{DATASET_NAME}",
            repo_id=DATASET_NAME,
            repo_type="dataset",
            allow_patterns=["votes/*"],
        )

    logger.info("Update metadata")
    for split in SPLITS:
        metadata = get_metadata(hf_api, split=split)

        df = pd.DataFrame.from_records(metadata)

        assert len(df["class"].unique()) == len(CLASS_CONCEPTS_VALUES.keys())

        for idx, class_ in zip(df.index, df["class"]):
            concepts = CLASS_CONCEPTS_VALUES[class_]
            for c in concepts:
                current_value = df.loc[idx, c]
                if current_value is None:
                    df.loc[idx, c] = True

        metadata = df.to_dict(orient="records")

        logger.info("Save metadata to Hub/Locally")
        save_metadata(hf_api, metadata, split, args.push_to_hub)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("pre-label-dataset")
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
