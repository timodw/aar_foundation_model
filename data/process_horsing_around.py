import argparse
from pathlib import Path
import csv
import pandas as pd
import numpy as np
import re
import h5py

from utils import dataset_to_hdf5

from typing import List, Dict, Tuple
from numpy.typing import NDArray


SAMPLING_RATE = 100 # in Hz
MOUNTING_LOCATION = 'neck'
SPECIES = 'horse'


def get_horse_mapping(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with open(path / 'subject_mapping.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        mapping = {int(row['subject']): row['name'] for row in reader}
            
    return mapping


def load_data_from_horse(
    path: Path, horse_id: int,
    columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'label']
) -> List[Dict[str, NDArray]]:
    loaded_datasets: List[Dict[str, NDArray]] = []
    for dataset_path in path.glob(f"subject_{horse_id}_*_part_*.csv"):
        p = re.compile('subject_[0-9]+_.+_part_([0-9]+).csv')
        result = p.search(dataset_path.name)
        dataset_id = int(result.group(1)) if result is not None else -1
        if dataset_id >= 0:
            df = pd.read_csv(dataset_path, low_memory=False)[columns]
            values_dict: Dict[str, NDArray] = {}
            if 'Ax' in columns:
                X_acc = df[['Ax', 'Ay', 'Az']].fillna(0.).values
                values_dict['acc'] = X_acc
            if 'Gx' in columns:
                X_gyr = df[['Gx', 'Gy', 'Gz']].fillna(0.).values
                values_dict['gyr'] = X_gyr
            if 'label' in columns:
                y = df['label'].fillna('unknown').values
                values_dict['label'] = y
            loaded_datasets.append(values_dict)

    return loaded_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/IDLab/horse_data/HorsingAround/data/csv', type=Path)
    parser.add_argument('--processed_path', default='/data/IDLab/aar_foundation_models/processed_data', type=Path)
    args = parser.parse_args()
    args.processed_path.mkdir(exist_ok=True, parents=True)

    horse_mapping = get_horse_mapping(args.csv_path)
    print(f"Found {len(horse_mapping)} horses.")
    
    for horse_id, horse_name in horse_mapping.items():
        loaded_datasets = load_data_from_horse(args.csv_path, horse_id)
        print(f"Loaded {len(loaded_datasets)} datasets for horse {horse_id} ({horse_name}).")
        for dataset_id, ds in enumerate(loaded_datasets):
            ds_path = dataset_to_hdf5(args.processed_path, horse_id, dataset_id, ds, SAMPLING_RATE, MOUNTING_LOCATION, SPECIES)
            print(f"Saved dataset {dataset_id} to {ds_path}.")
        print()