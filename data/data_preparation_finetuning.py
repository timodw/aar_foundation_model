import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from utils import get_all_individual_paths_for_dataset, hdf5_to_segments, individual_has_valid_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data/IDLab/aar_foundation_models/processed_data', type=Path)
    parser.add_argument('--output_folder', default='/data/IDLab/aar_foundation_models/training_snapshots/finetuning', type=Path)
    parser.add_argument('--dataset_name', default='horsing_around', type=str)
    parser.add_argument('--segment_duration', default=10., type=float)
    parser.add_argument('--max_window_length', default=1000, type=int)
    parser.add_argument('--n_folds', default=4, type=int)
    parser.add_argument('--random_seed', default=578, type=int)
    args = parser.parse_args()

    all_individual_paths = get_all_individual_paths_for_dataset(args.data_root / args.dataset_name)

    if not all_individual_paths:
        print(f"No data found for dataset '{args.dataset_name}' in {args.data_root / args.dataset_name}")
    else:
        print(f"Found {len(all_individual_paths)} total individuals for dataset '{args.dataset_name}'.")

        # Filter out individuals that only have 'unknown' labels
        labeled_individuals = [
            paths for paths in all_individual_paths 
            if individual_has_valid_labels(paths)
        ]

        print(f"Found {len(labeled_individuals)} individuals with valid labels.")

        if len(labeled_individuals) < args.n_folds:
            print(f"Error: Not enough labeled individuals ({len(labeled_individuals)}) to create {args.n_folds} folds.")
        else:
            kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_seed)
            individual_indices = np.arange(len(labeled_individuals))

            for fold, (train_idx, val_idx) in enumerate(kf.split(individual_indices)):
                print(f'--- Fold {fold+1}/{args.n_folds} ---')
                
                train_paths = [labeled_individuals[i] for i in train_idx]
                val_paths = [labeled_individuals[i] for i in val_idx]

                X_train, y_train = hdf5_to_segments(
                    train_paths,
                    segment_duration=args.segment_duration,
                    max_window_length=args.max_window_length,
                    shuffle=True,
                    random_seed=args.random_seed
                )

                X_val, y_val = hdf5_to_segments(
                    val_paths,
                    segment_duration=args.segment_duration,
                    max_window_length=args.max_window_length,
                    shuffle=False
                )

                print(f"Fold {fold+1}: {len(X_train)} training segments, {len(X_val)} validation segments.")

                output_dir = args.output_folder / args.dataset_name / f'fold_{fold+1}'
                output_dir.mkdir(parents=True, exist_ok=True)

                if X_train.size > 0:
                    np.save(output_dir / 'X_train.npy', X_train)
                    np.save(output_dir / 'y_train.npy', y_train)
                
                if X_val.size > 0:
                    np.save(output_dir / 'X_val.npy', X_val)
                    np.save(output_dir / 'y_val.npy', y_val)

            print(f'\nSuccessfully generated and saved {args.n_folds}-fold cross-validation datasets.')
