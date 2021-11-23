import argparse
import os
import shutil
from typing import List

import numpy as np
from numpy.random._generator import Generator
from tqdm import tqdm


def partition(
    rng: Generator, paths: List[str], train_proportion: float, val_proportion: float
):
    rng.shuffle(paths)
    n_train = int(round(len(paths) * train_proportion))
    n_val = int(round(len(paths) * val_proportion))

    train = paths[:n_train]
    val = paths[n_train : n_train + n_val]
    test = paths[n_train + n_val :]

    return train, val, test


def move_partition(
    filenames: List[str],
    src: str,
    dst: str,
    partition_type: str,
    attractor_class: str,
    copy=True,
):
    """
    :param filenames: list of files' filenames to copy
    :param src: source path
    :param dst: destination path
    :param partition_type: train, val or test
    :param attractor_class: regular or chaotic
    :param copy: if copy or move
    :return:
    """
    print(f"Moving {partition_type} {attractor_class}")

    dst_base_path = f"{dst}/{partition_type}/{attractor_class}"
    if not os.path.isdir(dst_base_path):
        os.makedirs(dst_base_path)

    for filename in tqdm(filenames):
        src_path = f"{src}/{filename}"
        dst_path = f"{dst_base_path}/{filename}"
        if copy:
            shutil.copy(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="global random generator seed"
    )
    parser.add_argument(
        "--train", type=float, default=0.8, help="proportion of training set"
    )
    parser.add_argument(
        "--val", type=float, default=0.1, help="proportion of validation set"
    )
    parser.add_argument("--src", type=str, default="Plots", help="source data folder")
    parser.add_argument(
        "--dst",
        type=str,
        default="partitioned_data",
        help="destination folder to save partitions",
    )
    rng = np.random.default_rng(parser.parse_args().seed)
    train_proportion = parser.parse_args().train
    val_proportion = parser.parse_args().val
    src_path = parser.parse_args().src
    dst_path = parser.parse_args().dst

    if train_proportion + val_proportion > 1.0:
        raise ValueError("Invalid partition, proportions add up to more than 1")

    chaotic_src = f"{src_path}/caotico/images"
    chaotic_img_paths = os.listdir(chaotic_src)
    regular_src = f"{src_path}/regular/images"
    regular_img_paths = os.listdir(regular_src)

    chaotic_train, chaotic_val, chaotic_test = partition(
        rng, chaotic_img_paths, train_proportion, val_proportion
    )
    regular_train, regular_val, regular_test = partition(
        rng, regular_img_paths, train_proportion, val_proportion
    )

    move_partition(chaotic_train, chaotic_src, dst_path, "train", "chaotic")
    move_partition(chaotic_val, chaotic_src, dst_path, "val", "chaotic")
    move_partition(chaotic_test, chaotic_src, dst_path, "test", "chaotic")

    move_partition(regular_train, regular_src, dst_path, "train", "regular")
    move_partition(regular_val, regular_src, dst_path, "val", "regular")
    move_partition(regular_test, regular_src, dst_path, "test", "regular")


if __name__ == "__main__":
    main()
