import argparse
from pathlib import Path
import pandas as pd
import re
from itertools import chain
from collections import defaultdict

from utils import dataset_to_hdf5
from process_waves_equine_activities import load_data_from_horse

from typing import Dict, List
from numpy.typing import NDArray


SAMPLING_RATE = 50 # in Hz
MOUNTING_LOCATION = 'head'
SPECIES = 'horse'


def get_dataset_paths(path: Path) -> Dict[int, List[Path]]:
    mapping: Dict[int, List[Path]] = defaultdict(list)
    for i, horse_path in enumerate(path.glob('*')):
        filtered_path = horse_path / 'filtered'
        if filtered_path.exists():
            for p in filtered_path.glob('*_head.csv'):
                mapping[i].append(p)
    
    return mapping




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/IDLab/foaling_detection/Datasets/foaling_2019/data', type=Path)
    parser.add_argument('--processed_path', default='/data/IDLab/aar_foundation_models/processed_data/idlab_foaling_2019', type=Path)
    args = parser.parse_args()
    args.processed_path.mkdir(exist_ok=True, parents=True)

    horse_ds_path_mapping = get_dataset_paths(args.csv_path)
    print(f"Found {len(horse_ds_path_mapping)} horses.")

    for horse_id, dataset_paths in horse_ds_path_mapping.items():
        loaded_datasets = load_data_from_horse(horse_id, dataset_paths,
                                              column_mapping={'acc': ['x', 'y', 'z']})
        print(f"Loaded {len(loaded_datasets)} dataset(s) for horse {horse_id}.")
        for dataset_id, ds in enumerate(loaded_datasets):
            ds_path = dataset_to_hdf5(args.processed_path, horse_id, dataset_id, ds, SAMPLING_RATE, MOUNTING_LOCATION, SPECIES)
            print(f"Saved dataset {dataset_id} to {ds_path}.")
    print()