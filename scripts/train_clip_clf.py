"""Simple classifier training script.

Run with:
```
poetry run python -m scripts.train_clip_clf
```
"""

import argparse

import torch
import wandb
from datasets import Features, Image, Value, load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import AutoConfig, CLIPForImageClassification, CLIPProcessor

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
class_feature = dataset["train"].features["class"]

processor = CLIPProcessor.from_pretrained(ARGS.model_name)
config = AutoConfig.from_pretrained(ARGS.model_name)
config.problem_type = "single_label_classification"
config.label2id = {
    label: str(i) for i, label in enumerate(class_feature.names)
}
config.id2label = {
    str(i): label for i, label in enumerate(class_feature.names)
}
config.num_labels = class_feature.num_classes

model = CLIPForImageClassification.from_pretrained(
    ARGS.model_name, config=config
)
model.to(DEVICE)
trainable_parameter_names = []
for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True
        trainable_parameter_names.append(name)
    else:
        param.requires_grad = False
print(f"[INFO] Trainable parameters: {trainable_parameter_names}")


def collate_fn(batch):
    images, classes = zip(*[(x["image"], x["class"]) for x in batch])
    return images, classes


train_dataloader = DataLoader(
    dataset["train"],
    batch_size=ARGS.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_dataloader = DataLoader(
    dataset["validation"],
    batch_size=ARGS.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

with wandb.init(  # type: ignore
    project="clip-clf",
    entity="mulsi",
    config={
        **vars(ARGS),
    },
) as wandb_run:
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
    for epoch in range(ARGS.n_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            images, classes = batch
            image_inputs = processor(
                images=images,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
            labels = torch.tensor(classes).to(DEVICE)
            optimizer.zero_grad()
            output = model(**image_inputs, labels=labels)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            wandb.log({"train/loss": loss.item()})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, batch in enumerate(val_dataloader):
                images, classes = batch
                image_inputs = processor(
                    images=images,
                    return_tensors="pt",
                )
                image_inputs = {
                    k: v.to(DEVICE) for k, v in image_inputs.items()
                }
                labels = torch.tensor(classes).to(DEVICE)
                output = model(**image_inputs, labels=labels)
                loss = output["loss"]
                val_loss += loss.item()
            val_loss /= len(val_dataloader)
            wandb.log({"val/loss": val_loss})

model.push_to_hub(
    ARGS.dataset_name.replace("concepts", ARGS.model_name.split("/")[-1]),
)
