"""Simple classifier training script.

Run with:
```
poetry run python -m scripts.train_clf
```
"""

import argparse
import pathlib
import shutil

import torch
from datasets import Features, Image, Value, load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

import wandb
from mulsi.hook import CacheHook, HookConfig
from scripts.constants import ASSETS_FOLDER, HF_TOKEN, WANDB_API_KEY

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
parser.add_argument("--n_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-5)
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
for param in model.parameters():
    param.requires_grad = False

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

cache_hook = CacheHook(HookConfig(module_exp=r".*\.layers\.\d+$"))
cache_hook.register(model.vision_model)
with torch.no_grad():
    for split, dataloader in dataloaders.items():
        for i, batch in enumerate(dataloader):
            images, ids = batch
            image_inputs = processor(
                images=images,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
            x = model.vision_model(**image_inputs)
            for layer, batched_activations in cache_hook.storage.items():
                folder = pathlib.Path(
                    f"{ASSETS_FOLDER}/data/"
                    f"{ARGS.model_name.replace('/', '.')}_{layer}/{split}"
                )
                folder.mkdir(parents=True, exist_ok=True)
                for i in range(ARGS.batch_size):
                    tensor = batched_activations[0][i]
                    t_id = ids[i]
                    torch.save(tensor, f"{folder}/{t_id}.pt")

            hf_api.upload_folder(
                folder_path=f"{ASSETS_FOLDER}/data",
                path_in_repo="data",
                repo_id=ARGS.dataset_name.replace("concepts", "activations"),
                repo_type="dataset",
            )
            shutil.rmtree(f"{ASSETS_FOLDER}/data")
