import argparse
import os

import joblib
from datasets import load_dataset
from loguru import logger
from sklearn.linear_model import LogisticRegression


def main(args):
    logger.info(f"Load dataset from {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    dataset = dataset.class_encode_column("class")

    train_ds = dataset["train"]
    test_ds = dataset["test"]
    logger.info(f"Train shape: {train_ds.shape}, Test shape: {test_ds.shape}")

    logger.info(f"Train LR classifier for max {args.max_iter} iterations")
    lr_clf = LogisticRegression(max_iter=args.max_iter)
    lr_clf.fit(X=train_ds["pooler"], y=train_ds["class"])

    score = lr_clf.score(X=test_ds["pooler"], y=test_ds["class"])
    logger.info(f"Accuracy score in test set: {score}")

    logger.info(f"Save model to {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    joblib.dump(lr_clf, os.path.join(args.output_dir, "lr_clf.joblib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="mulsi/fruit-vegetable-pooler"
    )
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
