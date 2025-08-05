from pathlib import Path
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

from typing import List, Dict
from numpy.typing import NDArray

from utils import (
    get_all_individual_paths_for_dataset,
    get_length_for_dataset,
    hdf5_to_segments
)


def main(args: argparse.Namespace) -> None:
    # Aggregate the paths of all datasets for each individual in these datasets
    pretraining_paths_per_ds: Dict[str, List[List[Path]]] = {}
    dataset_lengths: Dict[str, float] = {}
    for ds_name in args.datasets:
        paths_for_dataset: List[List[Path]] = get_all_individual_paths_for_dataset(args.data_root / ds_name)
        pretraining_paths_per_ds[ds_name] = paths_for_dataset
        ds_length: float = get_length_for_dataset(paths_for_dataset)
        dataset_lengths[ds_name] = ds_length
        print(f"Dataset '{ds_name}: {len(paths_for_dataset)} individuals ({int(ds_length):,} seconds).'")

    # Calculate the maximum allowed length for each dataset to ensure a fair distribution
    max_duration: float = min(dataset_lengths.values()) * args.oversampling_factor * args.max_dataset_imbalance
    print(f"Max duration per dataset: {int(max_duration):,} seconds", end='\n\n')

    training_arrays: List[NDArray] = []
    validation_arrays: List[NDArray] = []
    for ds_name in args.datasets:
        training_paths: List[List[Path]]
        validation_paths: List[List[Path]]
        training_paths, validation_paths = train_test_split(pretraining_paths_per_ds[ds_name],
                                                            train_size=args.train_ratio, random_state=args.random_seed)

        X_train: NDArray
        X_train, _ = hdf5_to_segments(
            training_paths,
            total_duration=min(dataset_lengths[ds_name] * args.oversampling_factor, max_duration),
            segment_duration=args.segment_duration,
            max_window_length=args.max_window_length,
            shuffle=True,
            random_seed=args.random_seed
        )

        X_val: NDArray
        X_val, _ = hdf5_to_segments(
            validation_paths,
            total_duration=min(dataset_lengths[ds_name] * args.oversampling_factor, max_duration),
            segment_duration=args.segment_duration,
            max_window_length=args.max_window_length,
        )
        print(f"Dataset '{ds_name}': {len(X_train)} training segments, {len(X_val)} validation segments.")

        training_arrays.append(X_train)
        validation_arrays.append(X_val)
    
    rng: np.random.Generator = np.random.default_rng(seed=args.random_seed)
    X_train_total: NDArray = np.concatenate(training_arrays, axis=0)
    rng.shuffle(X_train_total)

    X_val_total: NDArray = np.concatenate(validation_arrays, axis=0)
    
    print(f"\nTotal pretraining dataset: {len(X_train_total)} training segments, {len(X_val_total)} validation segments.")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    np.save(args.output_folder / 'X_train.npy', X_train_total)
    np.save(args.output_folder / 'X_val.npy', X_val_total)


if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        type=Path,
                        help='Path of the root folder where the processed datasets are stored.')
    parser.add_argument('--output_folder',
                        type=Path,
                        help='Path to the folder of where the store the processed pretraining numpy arrays.')
    parser.add_argument('--datasets', nargs='+',
                        type=str,
                        help='The names of the datasets used for pretraining.')
    parser.add_argument('--train_ratio', default=.5,
                        type=float,
                        help='Percentage of individual animals used for training (Range [0.0, 1.0]).')
    parser.add_argument('--max_dataset_imbalance', default=4.,
                        type=float,
                        help='The maximum imbalance between datasets allowed.')
    parser.add_argument('--oversampling_factor', default=5.,
                        type=float,
                        help='Factor on how many times to oversample each dataset.')
    parser.add_argument('--segment_duration', default=10.,
                        type=float,
                        help='Length in seconds of each extracted segment.')
    parser.add_argument('--max_window_length', default=1000,
                        type=int,
                        help='Maximum length in number of samples for the segments.')
    parser.add_argument('--random_seed', default=578, type=int)
    args: argparse.Namespace = parser.parse_args()

    main(args)