import pickle
import random
import argparse
from pathlib import Path

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-size', dest='val_size', help='Validation set size', default=10000, type=int)
    parser.add_argument('--data-dir', dest='data_dir', help='Data directory', default='data', type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    with open(data_dir.joinpath("data_train"), "rb") as f:
        full_data = pickle.load(f)

    data_size = len(full_data["images"])
    mask = np.zeros(data_size, dtype=bool)
    mask[random.sample(range(data_size), args.val_size)] = True

    with open(data_dir.joinpath('valid'), 'wb') as f:
        pickle.dump({
            "images": full_data["images"][mask],
            "labels": np.array(full_data["labels"])[mask].tolist(),
        }, f, pickle.HIGHEST_PROTOCOL)

    with open(data_dir.joinpath('train'), 'wb') as f:
        pickle.dump({
            "images": full_data["images"][~mask],
            "labels": np.array(full_data["labels"])[~mask].tolist(),
        }, f, pickle.HIGHEST_PROTOCOL)
