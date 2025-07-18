import h5py
from pathlib import Path
import numpy as np
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
import re

from typing import List, Dict, Set
from numpy.typing import NDArray


PRETRAINING_DATASETS = ['idlab_foaling_2019', 'waves_equine_activities']


def get_all_individual_paths_for_dataset(dataset_root: Path) -> List[List[Path]]:
    paths_dict: Dict[List[Path]] = defaultdict(list)
    for p in dataset_root.glob('*.hdf5'):
        re_p = re.compile('[a-z]+_([0-9]+)_ds_[0-9]+.hdf5')
        re_result = re_p.search(p.name)
        animal_id = int(re_result[1])
        paths_dict[animal_id].append(p)

    return [v for v in paths_dict.values()]


def get_length_for_dataset(paths_list: List[List[Path]]) -> float:
    total_seconds = 0.
    for paths in paths_list:
        for p in paths:
            with h5py.File(p, 'r') as f:
                sr = int(f.attrs['sr'])
                data_type = next(filter(lambda e: e.startswith('acc'), f.keys()))
                total_seconds += len(f[data_type]) / sr
    return total_seconds


def hdf5_to_ndarray_segmentation(hdf5_paths: List[List[Path]], total_duration: int,
                                 segment_duration: float, max_window_length: int, shuffle=True) -> NDArray:
    rng = np.random.default_rng()
    selected_segments: List[NDArray] = []
    n_individuals = len(training_paths)
    n_segments_per_individual = int((total_duration / segment_duration) / (n_individuals))
    for record_paths in training_paths:
        n_segments_per_record = n_segments_per_individual // len(record_paths)
        for record_path in record_paths:
            with h5py.File(record_path, 'r') as f:
                sr = int(f.attrs['sr'])
                acc_columns = list(filter(lambda e: e.startswith('acc'), f.keys()))
                record_length = len(f[acc_columns[0]])
                segment_n_samples = int(segment_duration * sr)
                segment_start_indices = rng.choice(record_length - segment_n_samples, n_segments_per_record)
                for start_i in segment_start_indices:
                    for column_name in acc_columns:
                        seg = f[column_name][start_i:start_i + segment_n_samples]
                        if seg.shape[0] < max_window_length:
                            seg = np.pad(seg, ((0, max_window_length - seg.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
                        selected_segments.append(seg)
    X_stacked = np.stack(selected_segments)
    if shuffle:
        rng.shuffle(X_stacked)
    return X_stacked
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data/IDLab/aar_foundation_models/processed_data',
                        type=Path,
                        help='Path of the root folder where the processed datasets are stored.')
    parser.add_argument('--output_folder', default='/data/IDLab/aar_foundation_models/training_snapshots/pretraining',
                        type=Path,
                        help='Path to the folder of where the store the processed pretraining numpy arrays.')
    parser.add_argument('--datasets', nargs='?', default='idlab_foaling_2019',
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
    args = parser.parse_args()

    pretraining_paths_per_ds: Dict[str, List[List[Path]]] = {}
    dataset_lengths: Dict[str, float] = {}
    for ds_name in PRETRAINING_DATASETS:
        paths_for_dataset = get_all_individual_paths_for_dataset(args.data_root / ds_name)
        pretraining_paths_per_ds[ds_name] = paths_for_dataset
        ds_length = get_length_for_dataset(paths_for_dataset)
        dataset_lengths[ds_name] = ds_length
        print(f"Dataset '{ds_name}: {len(paths_for_dataset)} individuals ({int(ds_length):,} seconds).'")

    max_duration: float = min(dataset_lengths.values()) * args.oversampling_factor * args.max_dataset_imbalance
    print(f"Max duration per dataset: {int(max_duration):,} seconds", end='\n\n')

    training_arrays: List[NDArray] = []
    validation_arrays: List[NDArray] = []
    for ds_name in PRETRAINING_DATASETS:
        training_paths, validation_paths = train_test_split(pretraining_paths_per_ds[ds_name],
                                                            train_size=args.train_ratio, random_state=args.random_seed)

        X_train = hdf5_to_ndarray_segmentation(
            training_paths,
            total_duration=min(dataset_lengths[ds_name] * args.oversampling_factor, max_duration),
            segment_duration=args.segment_duration,
            max_window_length=args.max_window_length
        )
        print(f"Dataset '{ds_name}': loaded {len(X_train)} training segments.")

        X_val = hdf5_to_ndarray_segmentation(
            validation_paths,
            total_duration=min(dataset_lengths[ds_name] * args.oversampling_factor, max_duration),
            segment_duration=args.segment_duration,
            max_window_length=args.max_window_length
        )
        print(f"Dataset '{ds_name}': loaded {len(X_val)} validation segments.")

        training_arrays.append(X_train)
        validation_arrays.append(X_val)
    # At which phase should the data be normalized/standardized?
    rng = np.random.default_rng(seed=args.random_seed)
    X_train_total = np.concat(training_arrays, axis=0)
    rng.shuffle(X_train_total)

    X_val_total = np.concat(validation_arrays, axis=0)
    rng.shuffle(X_val_total)

    print(f"\nTotal pretraining dataset: {len(X_train_total)} training segments, {len(X_val_total)} validation segments.")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    np.save(args.output_folder / 'X_train.npy', X_train_total)
    np.save(args.output_folder / 'X_val.npy', X_val_total)
