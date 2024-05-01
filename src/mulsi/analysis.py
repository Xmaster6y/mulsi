"""Module to analyse adversarial images."""

import re

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import pil_to_tensor

from mulsi.adversarial import AdversarialImage
from mulsi.hook import HookConfig, MeasureHook


def produce_adv_im(
    base_image,
    target_label,
    processor,
    base_model,
    clf_model,
    label2id,
    loss,
    probes,
    epsilon=2,
    n_iter=10,
    use_sign=True,
):
    adv_im = AdversarialImage(base_image=pil_to_tensor(base_image.convert("RGB")).float())

    storage = []
    module_exp = ".*\.(?P<layer_name>layers\.\d+)$"

    def data_fn(output, name, **kwargs):
        m = re.match(module_exp, name)
        layer_name = m["layer_name"]
        if layer_name not in probes:
            return None
        measures = {}
        for probe_name, probe in probes[layer_name].items():
            measures[probe_name] = torch.sigmoid(probe(output[0].squeeze(0).detach()))
        return measures

    cache_hook = MeasureHook(HookConfig(module_exp=module_exp, data_fn=data_fn))

    @torch.no_grad
    def callback_fn(adv):
        data = {}
        image_inputs = processor(
            images=adv,
            return_tensors="pt",
        )
        output = base_model(**image_inputs)
        pooler = output.pooler_output
        data["logits"] = clf_model(pooler).squeeze(0)
        for k, v in cache_hook.storage.items():
            data[k] = v
        storage.append(data)

    try:
        cache_hook.register(base_model)
        adv_im.fgsm_iter_(
            epsilon=epsilon,
            n_iter=n_iter,
            use_sign=use_sign,
            input_list=[
                (
                    loss,
                    {"labels": torch.tensor([int(label2id[target_label])])},
                    -1.0,
                ),
            ],
            callback_fn=callback_fn,
        )
    finally:
        cache_hook.remove()
    return adv_im, storage


def plot_logits(storage, label2id, label_ids=None, labels=None):
    if label_ids is None and labels is None:
        raise ValueError("You must specify label_ids or labels")
    if labels is not None:
        label_ids = [int(label2id[label]) for label in labels]
    logit_dict = {label2id[label_id]: [] for label_id in label_ids}
    for s in storage:
        for label_id in label_ids:
            logit_dict[label2id[label_id]].append(s["logits"][label_id])
    for label, logits in logit_dict.items():
        plt.plot(range(1, len(logits) + 1), logits, label=label)
    plt.legend()
    plt.ylabel("Logit")
    plt.xlabel("Adv step")
    plt.show()


def plot_mean_proba(storage, layer_names, concepts):
    mean_pred_dict = {f"{layer_name}/{concept}": [] for layer_name in layer_names for concept in concepts}
    std_pred_dict = {f"{layer_name}/{concept}": [] for layer_name in layer_names for concept in concepts}
    for s in storage:
        for curve_name in mean_pred_dict.keys():
            layer_name, concept = curve_name.split("/")
            mean_pred_dict[curve_name].append(s[f"vision_model.encoder.{layer_name}"][concept].mean())
            std_pred_dict[curve_name].append(s[f"vision_model.encoder.{layer_name}"][concept].std())
    for label in mean_pred_dict.keys():
        plt.errorbar(
            range(len(mean_pred_dict[label])),
            mean_pred_dict[label],
            yerr=std_pred_dict[label],
            label=label,
        )
    plt.legend()
    plt.ylabel("Mean concept proba")
    plt.xlabel("Adv step")
    plt.show()


def plot_cls_proba(storage, layer_names, concepts):
    pred_dict = {f"{layer_name}/{concept}": [] for layer_name in layer_names for concept in concepts}
    for s in storage:
        for curve_name in pred_dict.keys():
            layer_name, concept = curve_name.split("/")
            pred_dict[curve_name].append(s[f"vision_model.encoder.{layer_name}"][concept][0, 0])
    for label in pred_dict.keys():
        plt.plot(
            range(len(pred_dict[label])),
            torch.stack(pred_dict[label]),
            label=label,
        )
    plt.legend()
    plt.ylabel("CLS concept proba")
    plt.xlabel("Adv step")
    plt.show()


def plot_proba_heatmap(storage, layer_names, concepts):
    pred_dict = {f"{layer_name}/{concept}": [] for layer_name in layer_names for concept in concepts}
    for s in storage:
        for curve_name in pred_dict.keys():
            layer_name, concept = curve_name.split("/")
            pred_dict[curve_name].append(s[f"vision_model.encoder.{layer_name}"][concept][1:, 0])
    for label in pred_dict.keys():
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        fig.tight_layout()
        ax1 = plt.subplot(1, 3, 1)
        ax1.set_title(f"{label} (step: 0)")
        ax1.imshow(pred_dict[label][0].reshape(7, 7), cmap="hot")
        ax2 = plt.subplot(1, 3, 2)
        step = len(pred_dict[label]) // 2
        ax2.set_title(f"{label} (step: {step-1})")
        ax2.imshow(pred_dict[label][step - 1].reshape(7, 7), cmap="hot")
        ax3 = plt.subplot(1, 3, 3)
        step = len(pred_dict[label])
        ax3.set_title(f"{label} (step: {step-1})")
        ax3.imshow(pred_dict[label][step - 1].reshape(7, 7), cmap="hot")
        plt.show()
