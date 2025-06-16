import argparse
from pathlib import Path
import pandas as pd
import re
from itertools import chain

from utils import dataset_to_hdf5

from typing import Dict, List
from numpy.typing import NDArray


SAMPLING_RATE = 100 # in Hz
MOUNTING_LOCATION = 'neck'
SPECIES = 'goat'


def get_goat_paths(dataset_path: Path) -> Dict[int, Path]:
    goat_mapping: Dict[int, Path] = {}
    for i, p in enumerate(dataset_path.glob('[A-Z][0-9]')):
        goat_mapping[i] = p
    
    return goat_mapping


def load_dataset_from_goat(
    ds_path: Path,
    column_mapping={'acc': ['ax', 'ay', 'az'],
                    'acc_high': ['axhg', 'ayhg', 'azhg'],
                    'gyr': ['gx', 'gy', 'gz'],
                    'label': ['label']}
) -> Dict[str, NDArray]:
    values_dict: Dict[str, NDArray] = {}
    columns: List[str] = list(chain(*list(column_mapping.values())))
    for i, sensor_path in enumerate(ds_path.glob('pos_[A-Z].csv')):
        df = pd.read_csv(sensor_path, low_memory=False)[columns]
        for data_type, type_columns in column_mapping.items():
            fill_vall = 0. if data_type != 'label' else 'unknown'
            data = df[type_columns].fillna(fill_vall).values
            values_dict[f"{data_type}_{i}" if data_type != 'label' else data_type] = data
    
    return values_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/IDLab/goats_kamminga', type=Path)
    parser.add_argument('--processed_path', default='/data/IDLab/aar_foundation_models/processed_data/kamminga_goats', type=Path)
    args = parser.parse_args()
    args.processed_path.mkdir(exist_ok=True, parents=True)

    goat_ds_path_mapping = get_goat_paths(args.csv_path)
    print(f"Found {len(goat_ds_path_mapping)} goats.")

    for goat_id, ds_path in goat_ds_path_mapping.items():
        ds: Dict[str, NDArray] = load_dataset_from_goat(ds_path)
        ds_path: Path = dataset_to_hdf5(args.processed_path, goat_id, 0, ds, SAMPLING_RATE, MOUNTING_LOCATION, SPECIES)
        print(f"Saved dataset for goat {goat_id} to {ds_path}.")