"""Module to analyse adversarial images."""

import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def plot_logits(
    storage,
    label2id,
    id2label,
    label_ids=None,
    labels=None,
    title=None,
    save_to=None,
):
    if label_ids is None and labels is None:
        raise ValueError("You must specify label_ids or labels")
    if labels is not None:
        label_ids = [int(label2id[label]) for label in labels]
    logit_dict = {id2label[label_id]: [] for label_id in label_ids}
    for s in storage:
        for label_id in label_ids:
            logit_dict[id2label[label_id]].append(s["logits"][label_id])
    plt.figure()
    for label, logits in logit_dict.items():
        plt.plot(range(len(logits)), logits, label=label)
    plt.legend()
    plt.ylabel("Logit")
    plt.xlabel("Adv step")
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


def plot_mean_proba(
    storage,
    layer_names,
    concepts,
    title=None,
    save_to=None,
):
    mean_pred_dict = {f"{layer_name}/{concept}": [] for layer_name in layer_names for concept in concepts}
    std_pred_dict = {f"{layer_name}/{concept}": [] for layer_name in layer_names for concept in concepts}
    for s in storage:
        for curve_name in mean_pred_dict.keys():
            layer_name, concept = curve_name.split("/")
            mean_pred_dict[curve_name].append(s[f"vision_model.encoder.{layer_name}"][concept].mean())
            std_pred_dict[curve_name].append(s[f"vision_model.encoder.{layer_name}"][concept].std())
    plt.figure()
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
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


def plot_cls_proba(
    storage,
    layer_names,
    concepts,
    title=None,
    save_to=None,
):
    pred_dict = {f"{layer_name}/{concept}": [] for layer_name in layer_names for concept in concepts}
    for s in storage:
        for curve_name in pred_dict.keys():
            layer_name, concept = curve_name.split("/")
            pred_dict[curve_name].append(s[f"vision_model.encoder.{layer_name}"][concept][0, 0])
    plt.figure()
    for label in pred_dict.keys():
        plt.plot(
            range(len(pred_dict[label])),
            torch.stack(pred_dict[label]),
            label=label,
        )
    plt.legend()
    plt.ylabel("CLS concept proba")
    plt.xlabel("Adv step")
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


def plot_proba_heatmap(
    storage,
    layer_names,
    concepts,
    cmap="PuBuGn",
    title=None,
    save_to=None,
):
    pred_dict = {f"{layer_name}/{concept}": [] for layer_name in layer_names for concept in concepts}
    for s in storage:
        for curve_name in pred_dict.keys():
            layer_name, concept = curve_name.split("/")
            pred_dict[curve_name].append(s[f"vision_model.encoder.{layer_name}"][concept][1:, 0])
    for label in pred_dict.keys():
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        plt.title(title)
        fig.tight_layout()
        ax1 = plt.subplot(1, 3, 1)
        ax1.set_title(f"{label} (step: 0)")
        ax1.imshow(pred_dict[label][0].reshape(7, 7), cmap=cmap, vmin=0, vmax=1)
        ax2 = plt.subplot(1, 3, 2)
        step = len(pred_dict[label]) // 2
        ax2.set_title(f"{label} (step: {step})")
        ax2.imshow(pred_dict[label][step].reshape(7, 7), cmap=cmap, vmin=0, vmax=1)
        ax3 = plt.subplot(1, 3, 3)
        step = len(pred_dict[label])
        ax3.set_title(f"{label} (step: {step-1})")
        ax3.imshow(pred_dict[label][step - 1].reshape(7, 7), cmap=cmap, vmin=0, vmax=1)
        if save_to is not None:
            plt.savefig(f"{save_to}".replace("label", label.replace("/", "_")))
            plt.close()
        else:
            plt.show()


def plot_mean_proba_through_layers(
    storage,
    layer_names,
    concepts,
    step_indices,
    title=None,
    save_to=None,
):
    mean_pred_dict = {f"{idx}/{concept}": [] for idx in step_indices for concept in concepts}
    std_pred_dict = {f"{idx}/{concept}": [] for idx in step_indices for concept in concepts}
    for step_index in step_indices:
        s = storage[step_index]
        for layer_name in layer_names:
            for concept in concepts:
                label_name = f"{step_index}/{concept}"
                mean_pred_dict[label_name].append(s[f"vision_model.encoder.{layer_name}"][concept].mean())
                std_pred_dict[label_name].append(s[f"vision_model.encoder.{layer_name}"][concept].std())
    plt.figure()
    for label in mean_pred_dict.keys():
        plt.errorbar(
            range(len(mean_pred_dict[label])),
            mean_pred_dict[label],
            yerr=std_pred_dict[label],
            label=label,
        )
    plt.legend()
    plt.ylabel("Mean concept proba")
    plt.xlabel("layer")
    plt.xticks(range(len(layer_names)), [layer_name.split(".")[1] for layer_name in layer_names])
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


def plot_metric_boxes(
    data,
    title=None,
    save_to=None,
):
    labels = next(iter(data.values())).keys()
    boxed_data = list(zip(*[m.values() for m in data.values()]))
    plt.boxplot(boxed_data, notch=True, vert=True, patch_artist=True, labels=labels)
    plt.ylabel("Metric value")
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


def plot_metric_boxes_per_layer(
    metrics,
    concept,
    title=None,
    save_to=None,
):
    boxed_data = []
    tick_labels = [layer_name for layer_name in metrics.keys()]
    color_labels = ["precision", "recall", "f1"]
    for layer_metrics in metrics.values():
        data = layer_metrics[concept]["per_pixel"]
        boxed_data += list(zip(*[m.values() for m in data.values()]))

    positions = [[i - 0.75, i, i + 0.75] for i in range(2, len(tick_labels) * 3 + 2, 3)]
    positions = [item for sublist in positions for item in sublist]
    bplot = plt.boxplot(boxed_data, notch=True, vert=True, patch_artist=True, positions=positions)

    colors = ["pink", "lightblue", "lightgreen"]
    all_colors = colors * len(metrics)
    for patch, color in zip(bplot["boxes"], all_colors):
        patch.set_facecolor(color)

    handles = []
    for color, label in zip(colors, color_labels):
        handles.append(mpatches.Patch(color=color, label=label))

    plt.xlabel("layer")
    plt.xticks(range(2, len(tick_labels) * 3 + 2, 3), [layer_name.split(".")[1] for layer_name in tick_labels])
    plt.legend(handles=handles)
    plt.ylabel("Metric value")
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
