"""Script for pre-labeling the dataset.

Run with:
```
poetry run python -m scripts.pre_labeling_dataset
```
"""

import os
import json
import argparse

import jsonlines
from huggingface_hub import HfApi
from loguru import logger

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from scripts.constants import (
    ASSETS_FOLDER,
    CLASS_CONCEPTS_VALUES,
    DATASET_NAME,
    HF_TOKEN,
    SPLITS,
    CONCEPTS,
    LABELED_CLASSES,
)

from dotenv import load_dotenv

load_dotenv()

PROMPT = """\
Given an image and its class, provide the cnocepts that are present in the image in the following format:

You may choose from the following concepts only:
{concepts}

Provide the classification in the following format:
Classification:::
Concepts: (concept: e.g., red, sphere, stem, etc.)

Examples:
Image: {imaage_example_1}
Class: {class_example_1}
Concepts: {concepts_example_1}

Image: {imaage_example_2}
Class: {class_example_2}
Concepts: {concepts_example_2}

Now here is an image and its class:
Image: {image}
Class: {class_}

Classification:::
Concepts:
"""


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


def compute_concepts(votes):
    vote_sum = {c: 0 for c in CONCEPTS}
    for vote in votes.values():
        for c in CONCEPTS:
            if c not in vote:
                continue
            vote_sum[c] += 2 * vote[c] - 1
    return {c: vote_sum[c] > 0 if vote_sum[c] != 0 else None for c in CONCEPTS}


def main(args):
    hf_api = HfApi(token=HF_TOKEN)

    model = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    )

    logger.info("Download metadata and votes")
    metadata, votes = get_votes(hf_api)

    for split in SPLITS:
        for item in metadata[split]:
            if item["class"] in LABELED_CLASSES:
                continue
            key = item["id"]

            # Call VLM
            message = HumanMessage(
                content=PROMPT.format(concepts=", ".join(CONCEPTS), class_=item["class"]),
                temperature=0,
            )
            response = model.invoke([message])
            pred = response.content.split(":")[1].strip() if ":" in response.content else response.content
            print(pred)
            concepts = get_pre_labeled_concepts(item)
            if "imenelydiaker" not in votes[key]:
                continue
            votes[key] = {"imenelydiaker": concepts}

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

    new_metadata = {}
    for split in ["train", "test"]:
        new_metadata[split] = []
        with jsonlines.open(f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/metadata.jsonl") as reader:
            for row in reader:
                s_id = row["id"]
                if s_id in votes:
                    row.update(compute_concepts(votes[s_id]))
                new_metadata[split].append(row)
        with jsonlines.open(f"{ASSETS_FOLDER}/{DATASET_NAME}/data/{split}/metadata.jsonl", mode="w") as writer:
            writer.write_all(new_metadata[split])

    if args.push_to_hub:
        logger.info("Upload metadata to Hub")
        for split in SPLITS:
            save_metadata(hf_api, new_metadata[split], split, push_to_hub=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("pre-label-dataset")
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
