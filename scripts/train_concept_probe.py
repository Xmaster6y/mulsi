"""Create a probe using the activation of a CLIP model.

Run with:
```
poetry run python -m scripts.train_concept_probe
```
"""

import argparse

import einops
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mulsi.clf import CLF
from scripts.constants import ASSETS_FOLDER, HF_TOKEN


def main(args):
    logger.info(f"Load dataset from {args.dataset_name}")
    init_ds = load_dataset(args.dataset_name, args.config_name)
    filtered_ds = init_ds.filter(lambda s: s[args.concept] is not None)
    labeled_ds = filtered_ds.rename_column(args.concept, "label")
    labeled_ds = labeled_ds.class_encode_column("label")
    torch_ds = labeled_ds.with_format("torch")

    def map_fn(s_batched):
        b, p, h = s_batched["activation"].shape
        new_s_batched = {}
        new_s_batched["pixel_activation"] = einops.rearrange(
            s_batched["activation"], "b p h -> (b p) h"
        )
        new_s_batched["pixel_label"] = einops.repeat(
            s_batched["label"], "b -> (b p)", p=p
        )
        new_s_batched["pixel_index"] = einops.repeat(
            torch.arange(p), "p -> (b p)", b=b
        )
        return new_s_batched

    dataset = torch_ds.map(
        map_fn, remove_columns=["activation", "label"], batched=True
    )

    train_ds = concatenate_datasets([dataset["train"], dataset["validation"]])
    test_ds = dataset["test"]
    logger.info(f"Train shape: {train_ds.shape}, Test shape: {test_ds.shape}")

    logger.info("Grid Search for LR classifier")
    pipe_clf = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression())]
    )
    parameters = {"clf__max_iter": [200, 500], "clf__C": [1e-1, 1, 10]}
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
    gs = GridSearchCV(pipe_clf, parameters, scoring="f1", cv=sss, n_jobs=-1)

    logger.info("Train LR classifier")
    gs.fit(X=train_ds["pixel_activation"], y=train_ds["pixel_label"])
    logger.info(f"CV results: {gs.cv_results_}")
    best_clf = gs.best_estimator_
    score = best_clf.score(
        X=train_ds["pixel_activation"], y=train_ds["pixel_label"]
    )
    logger.info(f"Accuracy score in train set: {score}")

    score = best_clf.score(
        X=test_ds["pixel_activation"], y=test_ds["pixel_label"]
    )
    logger.info(f"Accuracy score in test set: {score}")

    logger.info(f"Save model to {ASSETS_FOLDER}")
    torch_clf = CLF(
        pipe_clf=best_clf, classes=labeled_ds["train"].features["label"].names
    )
    with open(ASSETS_FOLDER / "clf.pt", "wb") as f:
        torch.save(torch_clf, f)

    if args.push_to_hub:
        logger.info("Push model to Hugging Face Hub")
        hfapi = HfApi()
        hfapi.upload_file(
            repo_id=args.dataset_name.replace("activations", "probes"),
            path_or_fileobj=ASSETS_FOLDER / "clf.pt",
            path_in_repo=f"data/{args.config_name}/{args.concept}/clf.pt",
            token=HF_TOKEN,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="mulsi/fruit-vegetable-activations"
    )
    parser.add_argument("--config_name", type=str, default="layers.11")
    parser.add_argument("--concept", type=str, default="yellow")
    parser.add_argument(
        "--push_to_hub", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
