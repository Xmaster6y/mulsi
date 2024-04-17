"""Script to make a dataset of pooler outputs from a CLIP vision model.

Run with:
```
poetry run python -m scripts.make_pooler_dataset
```
"""

import argparse
import logging
import sys

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPVisionModel

from scripts.constants import HF_TOKEN

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    images, infos = [], []
    for x in batch:
        images.append(x.pop("image"))
        x.pop("original_name")
        infos.append(x)
    return images, infos


def make_batch_gen(
    batched_pooler,
    infos,
):
    def gen():
        for pooler, info in zip(batched_pooler, infos):
            yield {"pooler": pooler.detach().float().numpy(), **info}

    return gen


@torch.no_grad
def gen_pooler_from_model(
    processor,
    model,
    dataloader,
):
    for batch in dataloader:
        images, infos = batch
        image_inputs = processor(
            images=images,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
        out = model(**image_inputs)
        yield out["pooler_output"].detach().float().numpy(), infos


def main(args: argparse.Namespace):
    logging.info(f"Running on {DEVICE}")

    processor = CLIPImageProcessor.from_pretrained(args.model_name)
    model = CLIPVisionModel.from_pretrained(args.model_name)
    model.eval()
    model.to(DEVICE)

    dataset = load_dataset(args.dataset_name, revision="refs/convert/parquet")
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
        "pooler_output": DatasetDict(
            {
                split: Dataset.from_generator(
                    gen_pooler_from_model,
                    gen_kwargs={
                        "processor": processor,
                        "model": model,
                        "dataloader": dataloaders[split],
                    },
                )
                for split in splits
            }
        )
    }

    for config_name, dataset in dataset_dict.items():
        if args.push_to_hub:
            dataset.push_to_hub(
                repo_id=args.output_dataset_name,
                config_name=config_name,
                token=HF_TOKEN,
            )
        else:
            logging.info(f"Dataset {config_name}: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("train-clf")
    parser.add_argument(
        "--model_name", type=str, default="openai/clip-vit-base-patch32"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mulsi/fruit-vegetable-concepts",
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        default="mulsi/fruit-vegetable-pooler",
    )
    parser.add_argument(
        "--push_to_hub", argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
