import argparse
from pathlib import Path
import pandas as pd

from utils import dataset_to_hdf5

from typing import Dict, List
from numpy.typing import NDArray

 
SAMPLING_RATE = 25 # in Hz
MOUNTING_LOCATION = 'neck'
SPECIES = 'cow'


def get_datasets(
    df: pd.DataFrame, cow_id: int,
    column_mapping={'acc': ['accX', 'accY', 'accZ'],
                    'label': ['behaviour']}
) -> List[Dict[str, NDArray]]:
    datasets: List[Dict[str, NDArray]] = []

    cow_df = df[df.calfId == cow_id].sort_values('dateTime')
    for segment_id in cow_df.segId.unique():
        segment_df = cow_df[cow_df.segId == segment_id]
        values_dict: Dict[str, NDArray] = {}
        for data_type, type_columns in column_mapping.items():
            fill_val = 0. if data_type != 'label' else 'unknown'
            data = segment_df[type_columns].fillna(fill_val).values
            values_dict[data_type] = data
        datasets.append(values_dict)

    return datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/IDLab/Calfs/AcTBeCalf.csv', type=Path)
    parser.add_argument('--processed_path', default='/data/IDLab/aar_foundation_models/processed_data/calves', type=Path)
    args = parser.parse_args()
    args.processed_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.csv_path, low_memory=True)
    df.dateTime = pd.to_datetime(df.dateTime)
    cow_ids = df.calfId.unique().tolist()
    print(f"Found {len(cow_ids)} cows.")

    for cow_id in cow_ids:
        datasets = get_datasets(df, cow_id)
        print(f"Loaded {len(datasets)} datasets for cow {cow_id}.")
        for dataset_id, ds in enumerate(datasets):
            ds_path = dataset_to_hdf5(args.processed_path, cow_id, dataset_id, ds, SAMPLING_RATE, MOUNTING_LOCATION, SPECIES)
            print(f"Saved dataset {dataset_id} to {ds_path}.")
        print()