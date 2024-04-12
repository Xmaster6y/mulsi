"""Script to make a dataset of activations from a CLIP model.

Run with:
```
poetry run python -m scripts.make_activation_dataset
```
"""

import argparse
import re

import torch
from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

import wandb
from mulsi.hook import CacheHook, HookConfig
from scripts.constants import ASSETS_FOLDER, HF_TOKEN, WANDB_API_KEY
from scripts.utils.dataset import make_generators

####################
# HYPERPARAMETERS
####################
parser = argparse.ArgumentParser("train-clf")
parser.add_argument(
    "--model_name", type=str, default="openai/clip-vit-base-patch32"
)
parser.add_argument(
    "--dataset_name", type=str, default="Xmaster6y/fruit-vegetable-concepts"
)
parser.add_argument("--download_dataset", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--layers", type=str, default="0,6,12")
####################

ARGS = parser.parse_args()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on {DEVICE}")
hf_api = HfApi(token=HF_TOKEN)
wandb.login(key=WANDB_API_KEY)  # type: ignore

processor = CLIPProcessor.from_pretrained(ARGS.model_name)
model = CLIPModel.from_pretrained(ARGS.model_name)
model.eval()
model.to(DEVICE)
layers = ARGS.layers.split(",")

if ARGS.download_dataset:
    hf_api.snapshot_download(
        repo_id=ARGS.dataset_name,
        repo_type="dataset",
        local_dir=f"{ASSETS_FOLDER}/{ARGS.dataset_name}",
    )
features = Features({"image": Image(), "id": Value(dtype="string")})
dataset = load_dataset(
    f"{ASSETS_FOLDER}/{ARGS.dataset_name}", features=features
)


def collate_fn(batch):
    images, ids = zip(*[(x["image"], x["id"]) for x in batch])
    return images, ids


splits = ["train", "validation", "test"]
dataloaders = {
    split: DataLoader(
        dataset[split],
        batch_size=ARGS.batch_size,
        collate_fn=collate_fn,
    )
    for split in splits
}

cache_hook = CacheHook(
    HookConfig(module_exp=rf".*\.layers\.({'|'.join(layers)})$")
)
handles = cache_hook.register(model.vision_model)
print(f"[INFO] Registered {len(handles)} hooks")


def make_batch_gen(
    batched_activations,
    ids,
):
    def gen():
        for activation, act_id in zip(batched_activations, ids):
            yield {
                "activation": activation.cpu().float().numpy(),
                "id": act_id,
            }

    return gen


@torch.no_grad
def make_gen_list(
    gen_dict,
    dataloaders,
):
    module_exp = re.compile(r".*\.layers\.(?P<layer>\d+)$")
    for split, dataloader in dataloaders.items():
        for batch in dataloader:
            images, ids = batch
            image_inputs = processor(
                images=images,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
            model.vision_model(**image_inputs)
            for module, batched_activations in cache_hook.storage.items():
                m = module_exp.match(module)
                layer = m.group("layer")
                gen_dict[layer][split].append(
                    make_batch_gen(batched_activations, ids)
                )


gen_dict = make_generators(
    layers=layers,
    splits=splits,
    make_gen_list=make_gen_list,
)
ds = DatasetDict(
    {
        f"layers.{layer}": DatasetDict(
            {
                split: Dataset.from_generator(gen_dict[layer][split])
                for split in splits
            }
        )
        for layer in layers
    }
)

ds.push_to_hub(
    repo_id=ARGS.dataset_name.replace("concepts", "activations"),
)
