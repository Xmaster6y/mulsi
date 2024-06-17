"""Script for pre-labeling the dataset.

Run with:
```
poetry run python -m scripts.pre_labeling_dataset
```
"""

import os
import random
import json
import argparse
import base64
from io import BytesIO

from PIL import Image
import jsonlines
from huggingface_hub import HfApi
from loguru import logger
from openai import AzureOpenAI, ChatCompletion

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

class OpenAIRequest:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )
        self.concepts = ",".join(CONCEPTS)
    
    def __call__(self, item: dict, icl: dict, **kwargs) -> ChatCompletion:
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": """\
You are a helpful assistant that can help annotating images. Answer by giving the list of concepts you can see in the provided image.

Given an image and its class, provide the concepts that are present in the image.

You may choose from the following concepts only:
{self.concepts}

Provide the classification in the following JSON format:
{"red": True, "sphere": True, "stem": False, ...}
"""
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""\
Here is an image and its class:

Class: {icl["class"]}\nImage:
"""
                    },
                    {
                        "type": "image",
                        "image": icl["image"]
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Concepts: {icl['concepts']}"
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Now here is another image and its class, provide the concepts: \nClass: {item['class']}\nImage:"
                    },
                    {
                        "type": "image",
                        "image": item["image"]
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Concepts:"
                    }
                ]
            }
        ]

        return self.client.chat.completions.create(
            model="gpt-4o",
            messages=message,
            **kwargs
        )

def image2base64(image: BytesIO) -> str:
    return base64.b64encode(image.getvalue()).decode("utf-8")

def get_icl_example_dict(metadata: dict, split: str) -> dict:
    labeled_items_classes = ["tomato", "lemon", "kiwi", "lettuce", "cabbage", "paprika", "beetroots", "bell pepper"]
    labeled_items = [item for item in metadata[split] if item["class"] in labeled_items_classes]

    images = [item["image"] for item in labeled_items]
    classes = [item["class"] for item in labeled_items]
    concepts = [get_pre_labeled_concepts(item) for item in labeled_items] #TODO: remove and replace with correct function

    rand_idx = random.randint(0, len(labeled_items) - 1) # TODO: remove

    return {
        "class": classes[rand_idx],
        "image": image2base64(images[rand_idx]),
        "concepts": ",".join([c for c in concepts[rand_idx] if concepts[rand_idx][c]]),
    }   

def main(args):
    hf_api = HfApi(token=HF_TOKEN)

    logger.info("Download metadata and votes")
    metadata, votes = get_votes(hf_api)

    for split in SPLITS:
        for item in metadata[split]:
            if item["class"] in LABELED_CLASSES:
                continue
            key = item["id"]

            item_dict = {
                "class": item["class"],
                "image": image2base64(item["image"]), # TODO: fix open image
            }

            icl_dict = get_icl_example_dict(metadata=metadata, split=split)

            openai_request = OpenAIRequest()
            response = openai_request(  
                item=item_dict,
                icl=icl_dict, # TODO: build the ICL dict manually
                max_tokens=200,
                temperature=0,
            )
                
            pred = response.choices[0].message.content
            pred = pred[pred.rfind("{"):pred.rfind("}")]
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
    parser = argparse.ArgumentParser("auto-label-dataset")
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
