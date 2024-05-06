"""Script analyse FGSM adversarial images.

Run with:
```
poetry run python -m scripts.analysis.probes_mixed_checks
```
"""

import argparse
from typing import List
import os

import einops
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger

from sklearn.metrics import f1_score, recall_score, precision_score

from mulsi.adversarial import LRClfLoss
from scripts.constants import HF_TOKEN, ASSETS_FOLDER, LABELED_CLASSES, CLASSES
from mulsi import analysis

LAYER_NAMES = [f"layers.{i}" for i in range(12)]
CONCEPTS = ["yellow", "red", "sphere", "ovaloid", "stem", "cylinder", "pulp", "green"]
IDX = 0

hf_api = HfApi(token=HF_TOKEN)


def probe_single_eval(y_true, y_pred):
    metrics = {}
    metrics["precision"] = precision_score(y_true, y_pred)
    metrics["recall"] = recall_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred)
    return metrics


def eval_probe(
    probe: LRClfLoss,
    activation: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    classes: List[str],
    selected_classes,
):
    predictions = probe(activation) > 0
    global_metrics = probe_single_eval(labels, predictions)

    per_pixel_metrics = {}
    for i in range(50):
        bool_index = indices == i
        per_pixel_metrics[i] = probe_single_eval(labels[bool_index], predictions[bool_index])

    per_class_metrics = {}
    # for class_name in selected_classes:
    #     bool_index = torch.tensor([c == class_name for c in classes])
    #     per_class_metrics[class_name] = probe_single_eval(labels[bool_index], predictions[bool_index])

    return {"global": global_metrics, "per_pixel": per_pixel_metrics, "per_class": per_class_metrics}


def map_fn(s_batched):
    b, p, h = s_batched["activation"].shape
    new_s_batched = {}
    new_s_batched["pixel_activation"] = einops.rearrange(s_batched["activation"], "b p h -> (b p) h")
    new_s_batched["pixel_label"] = einops.repeat(s_batched["label"], "b -> (b p)", p=p)
    new_s_batched["pixel_class"] = [s_batched["class"][i] for i in range(b) for _ in range(p)]
    new_s_batched["pixel_index"] = einops.repeat(torch.arange(p), "p -> (b p)", b=b)
    return new_s_batched


def main(args: argparse.Namespace):
    dataset_name = args.dataset_name

    # Download probes dataset
    hf_api.snapshot_download(
        repo_id=dataset_name.replace("concepts", "probes"),
        repo_type="model",
        local_dir=ASSETS_FOLDER / dataset_name.replace("concepts", "probes"),
        revision=args.probe_ref,
    )

    subfolder = "only_labeled" if args.only_labeled else "all"
    os.makedirs(ASSETS_FOLDER / "figures" / "sanity_checks" / subfolder, exist_ok=True)
    metrics = {}
    for layer_name in LAYER_NAMES:
        metrics[layer_name] = {}

        # Download activations dataset
        ds_activations = load_dataset(
            args.dataset_name.replace("concepts", "activations"), split="test", name=layer_name
        )
        selected_classes = LABELED_CLASSES if args.only_labeled else CLASSES
        init_ds = ds_activations.filter(lambda s: s["class"] in selected_classes)

        for concept in CONCEPTS:
            filtered_ds = init_ds.filter(lambda s: s[concept] is not None)
            labeled_ds = filtered_ds.rename_column(concept, "label")
            labeled_ds = labeled_ds.class_encode_column("label")
            torch_ds = labeled_ds.select_columns(["activation", "label", "class"]).with_format("torch")
            pred_dataset = torch_ds.map(map_fn, remove_columns=["activation", "label", "class"], batched=True)

            with open(
                ASSETS_FOLDER
                / f"{dataset_name.replace('concepts', 'probes')}/data/{LAYER_NAMES[IDX]}/{concept}/clf.pt",
                "rb",
            ) as f:
                probe = torch.load(f)

            metrics[layer_name][concept] = eval_probe(
                probe,
                pred_dataset["pixel_activation"],
                pred_dataset["pixel_label"],
                pred_dataset["pixel_index"],
                pred_dataset["pixel_class"],
                selected_classes,
            )
            logger.info(
                f"Layer: {layer_name}, Concept: {concept}, Global metrics: {metrics[layer_name][concept]['global']}"
            )

    for concept in CONCEPTS:
        analysis.plot_metric_boxes_per_layer(
            metrics,
            concept,
            title=f"{LAYER_NAMES[IDX]}/{concept}",
            save_to=ASSETS_FOLDER
            / "figures"
            / "sanity_checks"
            / subfolder
            / f"{LAYER_NAMES[IDX]}_{concept}_mixed_boxes.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("probes-mixed-checks")
    parser.add_argument("--mode", type=str, default="torch_clf")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mulsi/fruit-vegetable-concepts",
    )
    parser.add_argument("--probe_ref", type=str, default=None)
    parser.add_argument("--only_labeled", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # Probe refs
    # 29c94861ed9922843d4821f23e7e44fbb30f2de4 -> 3 CLF pre-labeling
    # ? -> 12 CLF all post-labeling
    # ? -> 12 CLF only_labeled post-labeling
