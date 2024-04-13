"""Script to make a dataset of poolers from a CLIP model.

Run with:
```
poetry run python -m scripts.make_pooler_dataset
```
"""

import argparse

import torch
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

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

if ARGS.download_dataset:
    hf_api.snapshot_download(
        repo_id=ARGS.dataset_name,
        repo_type="dataset",
        local_dir=f"{ASSETS_FOLDER}/{ARGS.dataset_name}",
    )

dataset = load_dataset(f"{ASSETS_FOLDER}/{ARGS.dataset_name}")
print(f"[INFO] Loaded dataset: {dataset}")


def collate_fn(batch):
    images = []
    infos = []
    for x in batch:
        images.append(x.pop("image"))
        x.pop("original_name")
        infos.append(x)
    return images, infos


splits = ["train", "validation", "test"]
dataloaders = {
    split: DataLoader(
        dataset[split],
        batch_size=ARGS.batch_size,
        collate_fn=collate_fn,
    )
    for split in splits
}


def make_batch_gen(
    batched_pooler_output,
    infos,
):
    def gen():
        for pooler_output, info in zip(batched_pooler_output, infos):
            yield {
                "pooler_output": pooler_output.cpu().float().numpy(),
                **info,
            }

    return gen


@torch.no_grad
def make_gen_list(
    gen_dict,
    dataloaders,
):
    for split, dataloader in dataloaders.items():
        for batch in dataloader:
            images, infos = batch
            image_inputs = processor(
                images=images,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
            output = model.vision_model(**image_inputs)
            gen_dict[split].append(
                make_batch_gen(output.pooler_output.detach(), infos)
            )


gen_dict = make_generators(
    configs=None,
    splits=splits,
    make_gen_list=make_gen_list,
    dataloaders=dataloaders,
)
dataset = DatasetDict(
    {split: Dataset.from_generator(gen_dict[split]) for split in splits}
)

dataset.push_to_hub(
    repo_id=ARGS.dataset_name.replace("concepts", "poolers"),
    config_name=ARGS.model_name.replace("/", "__"),
)
