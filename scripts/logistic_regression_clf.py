import argparse
import os

import joblib
from datasets import concatenate_datasets, load_dataset
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main(args):
    logger.info(f"Load dataset from {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    dataset = dataset.class_encode_column("class")

    train_ds = concatenate_datasets([dataset["train"], dataset["validation"]])
    test_ds = dataset["test"]
    logger.info(f"Train shape: {train_ds.shape}, Test shape: {test_ds.shape}")

    logger.info("Grid Search for LR classifier")
    parameters = {"max_iter": [100, 500]}
    lr = LogisticRegression()
    lr_clf = GridSearchCV(lr, parameters)

    logger.info("Train LR classifier")
    pipe = Pipeline([("center", StandardScaler()), ("classify", lr_clf)])
    pipe.fit(X=train_ds["pooler"], y=train_ds["class"])

    score = pipe.score(X=train_ds["pooler"], y=train_ds["class"])
    logger.info(f"Accuracy score in train set: {score}")

    score = pipe.score(X=test_ds["pooler"], y=test_ds["class"])
    logger.info(f"Accuracy score in test set: {score}")

    logger.info(f"Save model to {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    joblib.dump(pipe, os.path.join(args.output_dir, "lr_clf.joblib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="mulsi/fruit-vegetable-pooler"
    )
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="assets")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
