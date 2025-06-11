import argparse
from pathlib import Path
import csv
import pandas as pd
import numpy as np
import re

from typing import List, Dict, Tuple
from numpy.typing import NDArray


def get_horse_mapping(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with open(path / 'subject_mapping.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        mapping = {int(row['subject']): row['name'] for row in reader}
            
    return mapping


def load_data_from_horse(
    path: Path, horse_id: int,
    columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'label']
) -> List[List[NDArray]]:
    loaded_datasets: List[List[NDArray]] = []
    for dataset_path in path.glob(f"subject_{horse_id}_*_part_*.csv"):
        p = re.compile('subject_[0-9]+_.+_part_([0-9]+).csv')
        result = p.search(dataset_path.name)
        dataset_id = int(result.group(1)) if result is not None else -1
        if dataset_id >= 0:
            df = pd.read_csv(dataset_path)[columns]
            if 'label' in columns:
                df['label'] = pd.Series(df['label']).fillna('unknown')


    return loaded_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/IDLab/horse_data/HorsingAround/data/csv', type=Path)
    args = parser.parse_args()

    horse_mapping = get_horse_mapping(args.csv_path)
    print(f"Found {len(horse_mapping)} horses!")


