import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset, concatenate_datasets
from matplotlib.ticker import MaxNLocator


def load_datasets_from_json(dataset_path: Path, filenames: list[str]):
    datasets = [
        Dataset.from_json(str(dataset_path / filename)) for filename in filenames
    ]
    return datasets


def plot_sentence_lengths_histogram(
    sentences: list[str],
    figure_path: Path,
    figure_title: str,
    divider: int,
    min_threshold: int,
    max_threshold: int,
    reverse_sort: bool = False,
    x_label: str = "Words count per sentence",
    y_label: str = "Number of sentences",
):
    word_count_groups = []
    for sentence in sentences:
        word_count = len(sentence.split())
        num_word_count_group = int(word_count / divider) + 1
        lower_group_boundary = divider * num_word_count_group - divider
        upper_group_boundary = divider * num_word_count_group - 1
        word_count_group = f"{lower_group_boundary}-{upper_group_boundary}"
        word_count_groups.append(word_count_group)

    counter = Counter(word_count_groups)
    filtered_counter = {
        x: count
        for x, count in counter.items()
        if min_threshold <= int(x.split("-")[0])
        and int(x.split("-")[1]) <= max_threshold
    }
    sorted_counter = sorted(
        filtered_counter.items(),
        key=lambda pair: int(pair[0].split("-")[0]),
        reverse=reverse_sort,
    )
    labels, values = zip(*sorted_counter)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    bars = plt.bar(labels, values, color="dimgray")
    plt.title(figure_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, height,
                 ha='center', va='bottom')

    _create_dirs_if_not_exists(figure_path.parent)

    plt.savefig(figure_path, dpi=300)
    plt.show()


def convert_to_tf_dataset(
    hf_dataset: Dataset,
    columns: list[str],
    label_cols: list[str] | None,
    collator,
    batch_size: int,
    shuffle: bool = False,
):
    return hf_dataset.to_tf_dataset(
        columns=columns,
        label_cols=label_cols,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def get_best_epoch(
    history: dict,
    metric: str = "val_loss",
    metric_evaluator: str = "min",
):
    allowed_metric_evaluators = ["min", "max"]

    if metric_evaluator in allowed_metric_evaluators:
        match metric_evaluator:
            case "min":
                return int(np.argmin(history[metric]) + 1)
            case "max":
                return int(np.argmax(history[metric]) + 1)
            case _:
                raise Exception("Wrong metric evaluator passed!")


def plot_and_save_fig_from_history(
    history: dict,
    attributes: list[str],
    title: str,
    y_label: str,
    x_label: str,
    legend_descriptors: list[str],
    figure_dir_path: Path,
    figure_filename: str,
):
    for attribute in attributes:
        metric = history[attribute]
        plt.plot(range(1, len(metric) + 1), metric)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(legend_descriptors, loc="upper left")

    _create_dirs_if_not_exists(figure_dir_path)

    plt.savefig(figure_dir_path / figure_filename)
    plt.show()


def save_dict_as_json(dictionary: dict, dir_path: Path, filename: str):
    if not dir_path.exists() or not dir_path.is_dir():
        dir_path.mkdir(parents=True)

    with open(dir_path / filename, "w") as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def read_json_as_dict(path: Path) -> dict:
    with open(path, "r") as fp:
        data = json.load(fp)
        return data


def get_gpu_name():
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        return details.get("device_name", "Unknown GPU")


def _create_dirs_if_not_exists(directory: Path):
    if not directory.is_dir():
        directory.mkdir(parents=True)
