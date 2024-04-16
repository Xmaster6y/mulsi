"""Script to make a dataset of pooler outputs from a CLIP vision model.

Run with:
```
poetry run python -m scripts.make_pooler_dataset
```
"""

import sys
import argparse
import logging
import re

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPImageProcessor

from scripts.constants import ASSETS_FOLDER, HF_TOKEN

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    images, labels = [], []
    for x in batch:
        images.append(x.pop("image"))
        labels.append(x.pop("class"))
    return images, labels


def make_batch_gen(
    batched_pooler,
    labels,
):
    def gen():
        for pooler, label in zip(batched_pooler, labels):
            yield {"pooler": pooler.detach().float().numpy(), "labels": label}

    return gen


@torch.no_grad
def gen_pooler_from_model(
    processor,
    model,
    dataloader,
):
    for batch in dataloader:
        images, labels = batch
        image_inputs = processor(
            images=images,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
        out = model(**image_inputs)
        yield out['pooler_output'].detach().float().numpy(), labels


def main(args: argparse.Namespace):
    logging.info(f"Running on {DEVICE}")
    hf_api = HfApi(token=HF_TOKEN)

    processor = CLIPImageProcessor.from_pretrained(args.model_name)
    model = CLIPVisionModel.from_pretrained(args.model_name)
    model.eval()
    model.to(DEVICE)

    if args.download_dataset:
        hf_api.snapshot_download(
            repo_id=args.dataset_name,
            repo_type="dataset",
            local_dir=f"{ASSETS_FOLDER}/{args.dataset_name}",
            revision="refs/convert/parquet"
        )

    dataset = load_dataset(
        args.dataset_name, 
        revision="refs/convert/parquet"
    )
    logging.info(f"Loaded dataset: {dataset}")

    splits = ["train", "validation", "test"]
    dataloaders = {
        split: DataLoader(
            dataset[split],
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )
        for split in splits
    }
    
    dataset_dict = {
        f"pooler_output": DatasetDict(
            {
                split: Dataset.from_generator(gen_pooler_from_model, 
                                              gen_kwargs={
                                                  "processor": processor, 
                                                  "model": model, 
                                                  "dataloader": dataloaders[split]
                                                })
                for split in splits
            }
        )
    }

    for dataset in dataset_dict.items():
        # dataset.push_to_hub(
        #     repo_id=args.output_dataset_name
        # )
        print(dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("train-clf")
    parser.add_argument(
        "--model_name", type=str, default="openai/clip-vit-base-patch32"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Xmaster6y/fruit-vegetable-concepts",
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        default="mulsi/fruit-vegetable-pooler"
    )
    parser.add_argument(
        "--download_dataset", action="store_true", default=False
    )
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
