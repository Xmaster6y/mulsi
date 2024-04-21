"""Script to make a dataset of different outputs from a CLIP vision model.

Run with:
```
poetry run python -m scripts.make_clf_dataset
```
"""

import argparse

import torch
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)

from scripts.constants import HF_TOKEN
from scripts.utils.dataset import merge_gens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    images, infos = [], []
    for x in batch:
        images.append(x.pop("image"))
        x.pop("original_name")
        infos.append(x)
    return images, infos


def make_batch_gen(
    batched_output,
    infos,
):
    def gen():
        for output, info in zip(batched_output, infos):
            yield {"output": output.cpu().float().numpy(), **info}

    return gen


@torch.no_grad
def gen_output_from_model(
    processor,
    model,
    dataloader,
    output_mode,
):
    gen_list = []
    for batch in dataloader:
        images, infos = batch
        image_inputs = processor(
            images=images,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
        out = model(**image_inputs)
        if output_mode == "pooler":
            output = out.pooler_output
        elif output_mode == "embeds":
            output = out.image_embeds
        elif output_mode == "mean_pooling":
            output = out.last_hidden_state[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Invalid output mode: {output_mode}")
        gen_list.append(make_batch_gen(output, infos))
    full_gen = merge_gens(gen_list)
    yield from full_gen()


def main(args: argparse.Namespace):
    logger.info(f"Running on {DEVICE}")

    processor = CLIPImageProcessor.from_pretrained(args.model_name)
    if args.output_mode == "embeds":
        model = CLIPVisionModelWithProjection.from_pretrained(args.model_name)
    else:
        model = CLIPVisionModel.from_pretrained(args.model_name)
    model.eval()
    model.to(DEVICE)

    dataset = load_dataset(args.dataset_name, revision="refs/convert/parquet")
    logger.info(f"Loaded dataset: {dataset}")

    splits = ["train", "validation", "test"]
    dataloaders = {
        split: DataLoader(
            dataset[split],
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )
        for split in splits
    }

    dataset_dict = DatasetDict(
        {
            split: Dataset.from_generator(
                gen_output_from_model,
                gen_kwargs={
                    "processor": processor,
                    "model": model,
                    "dataloader": dataloaders[split],
                    "output_mode": args.output_mode,
                },
            )
            for split in splits
        }
    )
    if args.push_to_hub:
        dataset_dict.push_to_hub(
            repo_id=args.output_dataset_name,
            config_name=args.output_mode,
            token=HF_TOKEN,
        )
    else:
        logger.info(f"Dataset {args.output_mode}: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("make-clf-dataset")
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
        default="mulsi/fruit-vegetable-outputs",
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        default="pooler",
        choices=["pooler", "embeds", "mean_pooling"],
    )
    parser.add_argument(
        "--push_to_hub", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
