import argparse
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict
from itertools import chain
import datetime

from utils import dataset_to_hdf5

from typing import Dict, List, Set
from numpy.typing import NDArray

 
SAMPLING_RATE = 40 # in Hz
MOUNTING_LOCATION = 'neck'
SPECIES = 'cat'

# AccX      AccY      AccZ Behaviour
def get_datasets(
    df: pd.DataFrame, cat_id: str,
    column_mapping={'acc': ['AccX', 'AccY', 'AccZ'],
                    'label': ['Behaviour']}
) -> List[Dict[str, NDArray]]:
    datasets: List[Dict[str, NDArray]] = []

    cat_df = df[df.ID == cat_id]
    groups = (cat_df.Behaviour != cat_df.Behaviour.shift()).cumsum()
    sub_dfs = [group for _, group in cat_df.groupby(groups)]
    for segment_df in sub_dfs:
        values_dict: Dict[str, NDArray] = {}
        for data_type, type_columns in column_mapping.items():
            fill_val = 0. if data_type != 'label' else 'unknown'
            data = segment_df[type_columns].fillna(fill_val).values
            values_dict[data_type] = data
        datasets.append(values_dict)

    return datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/IDLab/Cats/Dunford_et_al._Cats_calibrated_data.csv', type=Path)
    parser.add_argument('--processed_path', default='/data/IDLab/aar_foundation_models/processed_data/cats', type=Path)
    args = parser.parse_args()
    args.processed_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.csv_path, low_memory=True)
    cat_ids = df.ID.unique().tolist()
    print(f"Found {len(cat_ids)} cats.")

    for i, cat_id in enumerate(cat_ids):
        datasets = get_datasets(df, cat_id)
        print(f"Loaded {len(datasets)} dataset(s) for cat {cat_id}.")
        for dataset_id, ds in enumerate(datasets):
            ds_path = dataset_to_hdf5(args.processed_path, i, dataset_id, ds, SAMPLING_RATE, MOUNTING_LOCATION, SPECIES)
            print(f"Saved dataset {dataset_id} to {ds_path}.")
        print()