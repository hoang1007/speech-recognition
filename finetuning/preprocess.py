import sys

sys.path.append("..")

import os
import argparse
from torch.utils.data import random_split
from src.datamodule import VLSP2020TarDataset, VLSP2020Dataset


def prepare_tar_dataset(data_dir: str, dest_dir: str):
    dts = VLSP2020Dataset(data_dir)
    train_set, val_set = random_split(dts, [42_000, 14_427])

    VLSP2020TarDataset(os.path.join(dest_dir, "vlsp2020_train_set.tar")).convert(
        train_set
    )
    VLSP2020TarDataset(os.path.join(dest_dir, "vlsp2020_val_set.tar")).convert(val_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    args = parser.parse_args()

    prepare_tar_dataset(args.data_dir, args.dest_dir)
