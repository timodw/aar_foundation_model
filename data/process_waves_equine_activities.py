import argparse
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict
from itertools import chain

from utils import dataset_to_hdf5

from typing import Dict, List
from numpy.typing import NDArray

 
SAMPLING_RATE = 50 # in Hz
MOUNTING_LOCATION = 'legs'
SPECIES = 'horse'


def load_data_from_horse(
    horse_id: int, paths: List[Path],
    column_mapping={'acc_r': ['axR', 'ayR', 'azR'],
                    'acc_l': ['axL', 'ayL', 'azL'],
                    'label': ['label']}
) -> List[Dict[str, NDArray]]:
    loaded_datasets: List[Dict[str, NDArray]] = []
    columns = list(chain(*list(column_mapping.values())))
    for dataset_id, dataset_path in enumerate(paths):
        df = pd.read_csv(dataset_path, low_memory=False)[columns]
        values_dict: Dict[str, NDArray] = {}
        for data_type, type_columns in column_mapping.items():
            fill_val = 0. if data_type != 'label' else 'unknown'
            data = df[type_columns].fillna(fill_val).values
            values_dict[data_type] = data
        loaded_datasets.append(values_dict)

    return loaded_datasets


def get_horse_to_datasets_mapping(path: Path) -> Dict[int, List[Path]]:
    mapping = defaultdict(list)

    # Format 1
    format_1_horse_id_mapping: Dict[int, int] = {}
    current_id = 0
    for ds_path in path.glob('*annotated*.csv'):
        p = re.compile('[0-9]+-([0-9]+).*')
        result = p.search(ds_path.name)
        horse_id = int(result.group(1))

        if horse_id in format_1_horse_id_mapping:
            horse_id = format_1_horse_id_mapping[horse_id]
        else:
            format_1_horse_id_mapping[horse_id] = current_id
            horse_id = current_id
            current_id += 1

        mapping[horse_id].append(ds_path)
    
    # Format 2
    format_2_horse_id_mapping: Dict[int, int] = {}
    for ds_path in path.glob('Horse*.csv'):
        p = re.compile('Horse([0-9]+)')
        result = p.search(ds_path.name)
        horse_id = int(result.group(1))
        if horse_id in format_2_horse_id_mapping:
            horse_id = format_2_horse_id_mapping[horse_id]
        else:
            format_2_horse_id_mapping[horse_id] = current_id
            horse_id = current_id
            current_id += 1

        mapping[horse_id].append(ds_path)

    return mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/IDLab/horse_data/Activities', type=Path)
    parser.add_argument('--processed_path', default='/data/IDLab/aar_foundation_models/processed_data/waves_equine_activities', type=Path)
    args = parser.parse_args()
    args.processed_path.mkdir(exist_ok=True, parents=True)

    horse_ds_path_mapping = get_horse_to_datasets_mapping(args.csv_path)
    print(f"Found {len(horse_ds_path_mapping)} horses.")

    for horse_id, dataset_paths in horse_ds_path_mapping.items():
        loaded_datasets = load_data_from_horse(horse_id, dataset_paths)
        print(f"Loaded {len(loaded_datasets)} datasets for horse {horse_id}.")
        for dataset_id, ds in enumerate(loaded_datasets):
            ds_path = dataset_to_hdf5(args.processed_path, horse_id, dataset_id, ds, SAMPLING_RATE, MOUNTING_LOCATION, SPECIES)
            print(f"Saved dataset {dataset_id} to {ds_path}.")
        print()