"""Simple classifier training script.

Run with:
```
poetry run python -m scripts.train_clf
```
"""

import argparse

import torch
import wandb
from datasets import Features, Image, Value, load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from scripts.constants import ASSETS_FOLDER, HF_TOKEN, WANDB_API_KEY
from scripts.utils.clf import CLF

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
features = Features({"image": Image(), "class": Value(dtype="string")})
dataset = load_dataset(
    f"{ASSETS_FOLDER}/{ARGS.dataset_name}", features=features
)
dataset = dataset.class_encode_column("class")

train_dataloader = DataLoader(
    dataset["train"],
    batch_size=ARGS.batch_size,
    shuffle=True,
)
val_dataloader = DataLoader(
    dataset["validation"],
    batch_size=ARGS.batch_size,
    shuffle=False,
)

clf = CLF(
    n_hidden=model.config.hidden_size,
    classes=dataset["train"].features["class"].names,
)
clf.to(DEVICE)

with wandb.init(  # type: ignore
    project="mulsi-clf",
    config={
        **vars(ARGS),
    },
) as wandb_run:
    optimizer = torch.optim.Adam(clf.parameters(), lr=ARGS.lr)
    for epoch in range(ARGS.n_epochs):
        clf.train()
        for i, batch in enumerate(train_dataloader):
            inputs = processor(
                text=batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = batch["class"].to(DEVICE)
            optimizer.zero_grad()
            loss = clf.loss(model(**inputs).last_hidden_state, labels)
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})

        clf.eval()
        with torch.no_grad():
            val_loss = 0
            for i, batch in enumerate(val_dataloader):
                inputs = processor(
                    text=batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = batch["class"].to(DEVICE)
                loss = clf.loss(model(**inputs).last_hidden_state, labels)
                val_loss += loss.item()
            wandb.log({"val_loss": val_loss / len(val_dataloader)})

hf_api.upload_file(
    path_or_fileobj=f"{ASSETS_FOLDER}/model.pt",
    path_in_repo=f"data/{ARGS.model_name}",
    repo_id="Xmaster6y/fruit-vegetable-clfs",
    repo_type="dataset",
)
