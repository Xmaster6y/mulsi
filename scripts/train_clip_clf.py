"""Simple classifier training script.

Run with:
```
poetry run python -m scripts.train_clip_clf
```
"""

import argparse

import torch
import wandb
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoConfig, CLIPForImageClassification, CLIPProcessor

from scripts.constants import WANDB_API_KEY


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on {DEVICE}")
    wandb.login(key=WANDB_API_KEY)  # type: ignore

    dataset = load_dataset(
        args.dataset_name,
        revision="refs/convert/parquet",
    )
    dataset = dataset.class_encode_column("class")
    class_feature = dataset["train"].features["class"]

    processor = CLIPProcessor.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.problem_type = "single_label_classification"
    config.label2id = {
        label: str(i) for i, label in enumerate(class_feature.names)
    }
    config.id2label = {
        str(i): label for i, label in enumerate(class_feature.names)
    }
    config.num_labels = class_feature.num_classes

    model = CLIPForImageClassification.from_pretrained(
        args.model_name, config=config
    )
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)
    model.to(DEVICE)
    trainable_parameter_names = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            trainable_parameter_names.append(name)
        else:
            param.requires_grad = False
    logger.info(f"Trainable parameters: {trainable_parameter_names}")

    def collate_fn(batch):
        images, classes = zip(*[(x["image"], x["class"]) for x in batch])
        return images, classes

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    step = 0
    with wandb.init(  # type: ignore
        project="clip-clf",
        entity="mulsi",
        config={
            **vars(args),
        },
    ) as wandb_run:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.n_epochs):
            model.train()
            for batch in train_dataloader:
                step += 1
                images, classes = batch
                image_inputs = processor(
                    images=images,
                    return_tensors="pt",
                )
                image_inputs = {
                    k: v.to(DEVICE) for k, v in image_inputs.items()
                }
                labels = torch.tensor(classes).to(DEVICE)
                optimizer.zero_grad()
                output = model(**image_inputs, labels=labels)
                loss = output["loss"]
                loss.backward()
                optimizer.step()
                wandb_run.log({"train/loss": loss.item()}, step=step)
                logger.info(f"Step: {step}, Loss: {loss.item()}")

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
                wandb_run.log({"val/loss": val_loss}, step=step)

    if args.push_to_hub:
        model.push_to_hub(
            args.dataset_name.replace(
                "concepts", args.model_name.split("/")[-1]
            ),
        )
        processor.push_to_hub(
            args.dataset_name.replace(
                "concepts", args.model_name.split("/")[-1]
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("train-clip-clf")
    parser.add_argument(
        "--model_name", type=str, default="openai/clip-vit-base-patch32"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="mulsi/fruit-vegetable-concepts"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--push_to_hub", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
