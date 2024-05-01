"""Script to make a dataset of activations from a CLIP model.

Run with:
```
poetry run python -m scripts.make_activation_dataset
```
"""

import argparse
import re

import torch
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from loguru import logger
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from mulsi.hook import CacheHook, HookConfig
from scripts.constants import ASSETS_FOLDER, HF_TOKEN, WANDB_API_KEY
from scripts.utils.dataset import collate_fn, make_batch_gen, make_generators

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad
def make_gen_list(gen_dict, dataloaders, cache_hook, processor, model):
    module_exp = re.compile(r".*\.layers\.(?P<layer>\d+)$")
    for split, dataloader in dataloaders.items():
        for batch in dataloader:
            images, infos = batch
            image_inputs = processor(
                images=images,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
            model.vision_model(**image_inputs)
            for module, batched_activations in cache_hook.storage.items():
                m = module_exp.match(module)
                layer = m.group("layer")
                gen_dict[layer][split].append(make_batch_gen(batched_activations[0].detach(), infos, "activation"))


def main(args: argparse.Namespace):
    logger.info(f"Running on {DEVICE}")

    hf_api = HfApi(token=HF_TOKEN)
    wandb.login(key=WANDB_API_KEY)  # type: ignore

    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name)
    model.eval()
    model.to(DEVICE)

    if args.layers == "*":
        layers = [str(i) for i in range(model.vision_model.config.num_hidden_layers)]
    else:
        layers = args.layers.split(",")

    if args.download_dataset:
        hf_api.snapshot_download(
            repo_id=args.dataset_name,
            repo_type="dataset",
            local_dir=f"{ASSETS_FOLDER}/{args.dataset_name}",
        )

    dataset = load_dataset(f"{ASSETS_FOLDER}/{args.dataset_name}")
    print(f"[INFO] Loaded dataset: {dataset}")

    splits = ["train", "test"]
    dataloaders = {
        split: DataLoader(
            dataset[split],
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )
        for split in splits
    }

    cache_hook = CacheHook(HookConfig(module_exp=rf".*\.layers\.({'|'.join(layers)})$"))
    handles = cache_hook.register(model.vision_model)
    print(f"[INFO] Registered {len(handles)} hooks")

    gen_dict = make_generators(
        configs=layers,
        splits=splits,
        make_gen_list=make_gen_list,
        dataloaders=dataloaders,
        model=model,
        processor=processor,
        cache_hook=cache_hook,
    )
    config_dict = {
        f"layers.{layer}": DatasetDict({split: Dataset.from_generator(gen_dict[layer][split]) for split in splits})
        for layer in layers
    }

    if args.push_to_hub:
        for layer_name, dataset in config_dict.items():
            dataset.push_to_hub(
                repo_id=args.dataset_name.replace("concepts", "activations"),
                config_name=layer_name,
            )
    else:
        logger.info(f"Dataset {args.layers}: {config_dict}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("make-activation-dataset")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mulsi/fruit-vegetable-concepts",
    )
    parser.add_argument(
        "--download_dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--layers", type=str, default="0,6,11")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
