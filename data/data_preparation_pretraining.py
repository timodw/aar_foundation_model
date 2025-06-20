import h5py
from pathlib import Path
import numpy as np
import argparse
from collections import defaultdict
import re

from typing import List, Dict, Set


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data/IDLab/aar_foundation_models/processed_data', type=Path)
    parser.add_argument('--train_ratio', default=.5, type=float)
    parser.add_argument('--max_dataset_imbalance', default=10., type=float)
    parser.add_argument('--segment_duration', default=10., type=float)
    parser.add_argument('--max_window_length', default=1000, type=int)
    args = parser.parse_args()

    # Get total length of each dataset T (Dict[str, List[List[Path]]])
    # Calculate max allowed length for each dataset based on the ratio
    # Divide each dataset's individuals according to train ratio
    # Add random datasets from the individuals population until max length is reached

    pretraining_paths_per_ds: Dict[str, List[List[Path]]] = {}
    dataset_lengths: List[float] = []
    for ds_name in PRETRAINING_DATASETS:
        paths_for_dataset = get_all_individual_paths_for_dataset(args.data_root / ds_name)
        pretraining_paths_per_ds[ds_name] = paths_for_dataset
        ds_length = get_length_for_dataset(paths_for_dataset)
        dataset_lengths.append(ds_length)
        print(f"Dataset '{ds_name}: {len(paths_for_dataset)} individuals ({int(ds_length):,} seconds).'")

    max_duration: float = min(dataset_lengths) * args.max_dataset_imbalance
    print(f"Max duration per dataset: {int(max_duration):,} seconds", end='\n\n')
    