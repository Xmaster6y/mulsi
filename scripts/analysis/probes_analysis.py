"""Script analyse FGSM adversarial images.

Run with:
```
poetry run python -m scripts.analysis.probes_analysis
```
"""

import argparse

import einops
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger

from sklearn.metrics import f1_score, recall_score, precision_score

from mulsi.adversarial import LRClfLoss
from scripts.constants import HF_TOKEN, ASSETS_FOLDER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_NAMES = ["layers.0", "layers.6", "layers.11"]
CONCEPTS = ["yellow", "red", "sphere", "ovaloid"]
GOOD_INDICES = {
    "banana": [],  # None for all
    "lemon": [0, 6, 8],
    "tomato": [],
}

hf_api = HfApi(token=HF_TOKEN)


def eval_probe(probe: LRClfLoss, inputs: torch.Tensor, targets: list[str]):
    # TODO: impelment this for each pixel activation
    predictions = probe(inputs) > 0
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    return precision, recall, f1


def map_fn(s_batched):
    b, p, h = s_batched["activation"].shape
    new_s_batched = {}
    new_s_batched["pixel_activation"] = einops.rearrange(s_batched["activation"], "b p h -> (b p) h")
    new_s_batched["pixel_label"] = einops.repeat(s_batched["label"], "b -> (b p)", p=p)
    new_s_batched["pixel_index"] = einops.repeat(torch.arange(p), "p -> (b p)", b=b)
    return new_s_batched


def main(args: argparse.Namespace):
    logger.info(f"Running on {DEVICE}")
    dataset_name = args.dataset_name

    # Download probes dataset
    hf_api.snapshot_download(
        repo_id=dataset_name.replace("concepts", "probes"),
        repo_type="model",
        local_dir=ASSETS_FOLDER / dataset_name.replace("concepts", "probes"),
        revision=args.probe_ref,
    )

    probes, metrics = {}, {}
    for layer_name in LAYER_NAMES:
        probes[layer_name] = {}
        metrics[layer_name] = {}

        # Download activations dataset
        ds_activations = load_dataset(
            args.dataset_name.replace("concepts", "activations"), split="test", name=layer_name
        )

        for concept in CONCEPTS:
            filtered_ds = ds_activations.filter(lambda s: s[concept] is not None)
            labeled_ds = filtered_ds.rename_column(concept, "label")
            labeled_ds = labeled_ds.class_encode_column("label")
            torch_ds = labeled_ds.select_columns(["activation", "label"]).with_format("torch")
            pre_dataset = torch_ds.map(map_fn, remove_columns=["activation", "label"], batched=True)

            with open(
                ASSETS_FOLDER / f"{dataset_name.replace('concepts', 'probes')}/data/{layer_name}/{concept}/clf.pt",
                "rb",
            ) as f:
                probes[layer_name][concept] = torch.load(f)

            precision, recall, f1 = eval_probe(
                probes[layer_name][concept],
                pre_dataset["pixel_activation"],
                pre_dataset["pixel_label"],
            )
            metrics[layer_name][concept] = {
                metric_name: value
                for metric_name, value in zip(["precision", "recall", "f1"], [precision, recall, f1])
            }
            logger.info(f"Layer: {layer_name}, Concept: {concept}, Metrics: {metrics[layer_name][concept]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("fgsm-probing")
    parser.add_argument("--mode", type=str, default="torch_clf")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mulsi/fruit-vegetable-concepts",
    )
    parser.add_argument("--probe_ref", type=str, default=None)
    parser.add_argument("--epsilon", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--concept", type=str, default="yellow")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # Probe refs
    # 29c94861ed9922843d4821f23e7e44fbb30f2de4 -> 3 CLF pre-labeling
    # ? -> 12 CLF all post-labeling
    # ? -> 12 CLF only_labeled post-labeling
