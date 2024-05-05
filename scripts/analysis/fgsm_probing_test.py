"""Script to analyse FGSM adversarial images.

Run with:
```
poetry run python -m scripts.analysis.fgsm_probing_test
```
"""

import argparse
import os

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPVisionModel, AutoProcessor
from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger

from mulsi.preprocess import DiffCLIPImageProcessor
from mulsi import analysis
from mulsi.adversarial import LRClfLoss
from scripts.constants import HF_TOKEN, ASSETS_FOLDER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_NAMES = [f"layers.{i}" for i in range(12)]
CONCEPTS = ["yellow", "red", "sphere", "ovaloid", "stem", "cylinder", "pulp"]
GOOD_INDICES = {
    "banana": [1],
    "lemon": [0, 6, 8],
    "tomato": [0],
}

hf_api = HfApi(token=HF_TOKEN)


def setup_torch_clf(dataset_name):
    hf_api.snapshot_download(
        repo_id=dataset_name.replace("concepts", "clfs"),
        repo_type="model",
        local_dir=ASSETS_FOLDER / dataset_name.replace("concepts", "clfs"),
    )
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    with open(
        ASSETS_FOLDER
        / f"{dataset_name.replace('concepts', 'clfs')}/{dataset_name.replace('concepts', 'pooler')}/clf.pt",
        "rb",
    ) as f:
        clf_model = torch.load(f)
    image_processor = DiffCLIPImageProcessor(processor.image_processor)
    loss = LRClfLoss(clf_model=clf_model, base_model=base_model, image_processor=image_processor)
    id2label = clf_model.id2label
    label2id = clf_model.label2id
    return image_processor, base_model, clf_model, loss, id2label, label2id


def main(args: argparse.Namespace):
    logger.info(f"Running on {DEVICE}")
    dataset_name = args.dataset_name
    hf_api.snapshot_download(
        repo_id=dataset_name.replace("concepts", "probes"),
        repo_type="model",
        local_dir=ASSETS_FOLDER / dataset_name.replace("concepts", "probes"),
        revision=args.probe_ref,
    )
    dataset = load_dataset(dataset_name, split="test", revision="refs/convert/parquet")

    probes = {}
    for layer_name in LAYER_NAMES:
        probes[layer_name] = {}
        for concept in CONCEPTS:
            with open(
                ASSETS_FOLDER / f"{dataset_name.replace('concepts', 'probes')}/data/{layer_name}/{concept}/clf.pt",
                "rb",
            ) as f:
                probes[layer_name][concept] = torch.load(f)

    processor, base_model, clf_model, loss, id2label, label2id = setup_torch_clf(dataset_name)

    os.makedirs(ASSETS_FOLDER / "figures", exist_ok=True)
    for class_name, good_indices in GOOD_INDICES.items():
        filtered_ds = dataset.filter(lambda s: s["class"] == class_name)
        for target in GOOD_INDICES.keys():
            if target == class_name:
                continue
            os.makedirs(ASSETS_FOLDER / "figures" / f"{class_name}_{target}", exist_ok=True)
            it = range(len(filtered_ds)) if good_indices is None else good_indices
            for i in it:
                adv_im, storage = analysis.produce_adv_im(
                    filtered_ds[i]["image"],
                    target,
                    processor,
                    base_model,
                    clf_model,
                    label2id,
                    loss,
                    probes,
                    epsilon=args.epsilon,
                    n_iter=args.n_iter,
                    use_sign=True,
                )
                for concept in CONCEPTS:
                    analysis.plot_mean_proba_through_layers(
                        storage,
                        LAYER_NAMES,
                        [concept],
                        [0, args.n_iter],
                        title=f"{class_name} -> {target}",
                        save_to=ASSETS_FOLDER
                        / "figures"
                        / f"{class_name}_{target}"
                        / f"{i}_{concept}_through_layers.png",
                    )
                    analysis.plot_cls_proba(
                        storage,
                        LAYER_NAMES,
                        [concept],
                        title=f"{class_name} -> {target}",
                        save_to=ASSETS_FOLDER / "figures" / f"{class_name}_{target}" / f"{i}_{concept}_proba.png",
                    )
                    analysis.plot_mean_proba(
                        storage,
                        LAYER_NAMES,
                        [concept],
                        title=f"{class_name} -> {target}",
                        save_to=ASSETS_FOLDER / "figures" / f"{class_name}_{target}" / f"{i}_{concept}_mean_proba.png",
                    )
                analysis.plot_logits(
                    storage,
                    label2id,
                    id2label,
                    labels=["tomato", "lemon", "orange", "apple", "banana"],
                    title=f"{class_name} -> {target}",
                    save_to=ASSETS_FOLDER / "figures" / f"{class_name}_{target}" / f"{i}_logits.png",
                )
                # analysis.plot_proba_heatmap(
                #     storage,
                #     LAYER_NAMES,
                #     CONCEPTS,
                #     title=f"{class_name} -> {target}",
                #     save_to=ASSETS_FOLDER / "figures" / f"{class_name}_{target}" / f"{i}_heatmap_label.png",
                # )
                adv_pil_im = to_pil_image(adv_im.adv.cpu())
                adv_pil_im.save(ASSETS_FOLDER / "figures" / f"{class_name}_{target}" / f"{i}_adv.png")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # Probe refs
    # 29c94861ed9922843d4821f23e7e44fbb30f2de4 -> 3 CLF pre-labeling
    # ? -> 12 CLF all post-labeling
    # ? -> 12 CLF only_labeled post-labeling
