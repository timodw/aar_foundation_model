import h5py
from pathlib import Path
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from numpy.typing import NDArray


def dataset_to_hdf5(processed_data_path: Path, individual_id: int, dataset_id: int,
                    dataset: Dict[str, NDArray],
                    sampling_rate: int, mounting_location: str, species: str) -> Path:
    """
    Converts and stores a contiguous recording of sensor data to an HDF5 file.

    Args:
        processed_data_path (Path): The path in which the resulting HDF5 file 
        should be stored. File will be stored as 
        "{species}_{individual_id}_ds_{dataset_id}.hdf5".

        individual_id (int): The numerical ID identifying the individual animal.

        dataset_id (int): The ID of this specific recording of this specific
        animal.

        dataset (Dict[str, NDArray]): A dictionary containing a mapping from the
        name/type of the sensor (i.e. 'acc_left' or 'gyr_0') to the Numpy array
        containing the sensor data, for accelerometer data the prefix should
        always be 'acc'. Shape of the sensor data should be (n_samples, n_axes).
        For labeled data this dictionary should also contain a key 'label'
        containing the label for each sample.

        sampling_rate (int): The sensor sampling rate used to capture the data 
        (in Hz).

        mounting_location (str): The location of the sensor (i.e. neck, legs).

        species (str): The species of animal the data was captured
        from (i.e. horse).

    Returns:
        Path: The path of the created HDF5 file.
    """
    ds_path = processed_data_path / f"{species.lower()}_{individual_id}_ds_{dataset_id}.hdf5"
    with h5py.File(ds_path, 'w') as f:
        f.attrs['sr'] = sampling_rate
        f.attrs['loc'] = mounting_location
        f.attrs['species'] = species
        for key, data in dataset.items():
            ds = f.create_dataset(key, data.shape, dtype=data.dtype if data.dtype != np.dtypes.ObjectDType else h5py.string_dtype())
            ds[:] = data
    return ds_path


def get_all_individual_paths_for_dataset(dataset_root: Path) -> List[List[Path]]:
    """
    Groups HDF5 file paths by individual ID from a given dataset directory.

    The function assumes a file naming convention where the individual ID is
    a number following an underscore, e.g., 'horse_1_...'.
    """
    paths_dict: Dict[int, List[Path]] = defaultdict(list)
    for p in dataset_root.glob('*.hdf5'):
        re_p = re.compile(r'[a-z]+_(\d+)_ds_\d+.hdf5')
        re_result = re_p.search(p.name)
        if re_result:
            animal_id = int(re_result.group(1))
            paths_dict[animal_id].append(p)

    return list(paths_dict.values())


def get_length_for_dataset(paths_list: List[List[Path]]) -> float:
    """
    Calculates the total duration in seconds of all recordings in a list of path lists.
    """
    total_seconds = 0.
    for paths in paths_list:
        for p in paths:
            with h5py.File(p, 'r') as f:
                if 'sr' in f.attrs:
                    sr = int(f.attrs['sr'])
                    data_type = next(filter(lambda e: e.startswith('acc'), f.keys()), None)
                    if data_type and data_type in f and f[data_type].shape[0] > 0:
                        total_seconds += len(f[data_type]) / sr
    return total_seconds


def hdf5_to_segments(
    individual_paths: List[List[Path]],
    segment_duration: float,
    max_window_length: int,
    total_duration: Optional[float] = None,
    shuffle: bool = True,
    random_seed: int = None
) -> Tuple[NDArray, Optional[NDArray]]:
    """
    Extracts segments from HDF5 files.

    For fine-tuning (total_duration=None): extracts non-overlapping segments 
    from contiguous blocks of the same label.
    For pre-training (total_duration is set): extracts segments randomly.
    """
    rng = np.random.default_rng(random_seed)
    segments = []
    labels = []
    
    if total_duration:  # For pre-training style random sampling
        n_individuals = len(individual_paths)
        if n_individuals == 0:
            return np.array([]), None
        n_segments_per_individual = int((total_duration / segment_duration) / n_individuals) if segment_duration > 0 else 0

        for record_paths in individual_paths:
            if record_paths:
                n_segments_per_record = n_segments_per_individual // len(record_paths)
                for record_path in record_paths:
                    try:
                        with h5py.File(record_path, 'r') as f:
                            if 'sr' not in f.attrs: continue
                            sr = int(f.attrs['sr'])
                            acc_columns = sorted([k for k in f.keys() if k.startswith('acc')])
                            if not acc_columns or acc_columns[0] not in f: continue
                            record_length = len(f[acc_columns[0]])
                            segment_n_samples = int(segment_duration * sr)
                            if segment_n_samples <= 0 or record_length < segment_n_samples: continue
                            
                            start_indices = rng.choice(record_length - segment_n_samples, n_segments_per_record)
                            for start_i in start_indices:
                                end_i = start_i + segment_n_samples
                                seg_data_list = []
                                for col in acc_columns:
                                    if col in f:
                                        data = f[col][start_i:end_i]
                                        if data.ndim == 1: data = data[:, np.newaxis]
                                        seg_data_list.append(data)
                                
                                if not seg_data_list: continue
                                seg_data = np.concatenate(seg_data_list, axis=1)
                                if seg_data.shape[0] < max_window_length:
                                    pad_width = ((0, max_window_length - seg_data.shape[0]), (0, 0))
                                    seg_data = np.pad(seg_data, pad_width, mode='constant', constant_values=np.nan)
                                
                                seg_data = seg_data[:max_window_length, :]
                                segments.append(seg_data)
                    except Exception:
                        continue

    else:  # For fine-tuning, single-label, non-overlapping segments
        for record_paths in individual_paths:
            for record_path in record_paths:
                try:
                    with h5py.File(record_path, 'r') as f:
                        sr = int(f.attrs['sr'])
                        segment_n_samples = int(segment_duration * sr)
                        acc_columns = sorted([k for k in f.keys() if k.startswith('acc')])

                        record_length = len(f[acc_columns[0]])

                        if record_length < segment_n_samples:
                            continue

                        all_labels_raw = f['label'][:]
                        if all_labels_raw.dtype.kind == 'S':
                            all_labels = np.char.decode(all_labels_raw, 'utf-8')
                        else:
                            all_labels = all_labels_raw
                    
                        if len(all_labels) == 0:
                            continue

                        for start_i in range(0, record_length - segment_n_samples + 1, segment_n_samples):
                            end_i = start_i + segment_n_samples
                            
                            window_labels = all_labels[start_i:end_i]
                            first_label = window_labels[0]
                            if first_label == b'unknown' or not np.all(window_labels == first_label):
                                continue
                            
                            seg_data_list = []
                            for col in acc_columns:
                                data = f[col][start_i:end_i]
                                if data.ndim == 1:
                                    data = data[:, np.newaxis]
                                seg_data_list.append(data)
                            
                            if not seg_data_list:
                                continue

                            seg_data = np.concatenate(seg_data_list, axis=1)

                            if seg_data.shape[0] < max_window_length:
                                pad_width = ((0, max_window_length - seg_data.shape[0]), (0, 0))
                                seg_data = np.pad(seg_data, pad_width, mode='constant', constant_values=np.nan)
                            
                            segments.append(seg_data)
                            labels.append(first_label)
                except Exception:
                    continue

    X_stacked = np.stack(segments)
    
    y_stacked = None
    if not total_duration:
        y_stacked = np.array(labels, dtype=object)

    if shuffle:
        perm = rng.permutation(len(X_stacked))
        X_stacked = X_stacked[perm]
        if y_stacked is not None:
            y_stacked = y_stacked[perm]

    if total_duration:
        return X_stacked, None
    return X_stacked, y_stacked


def individual_has_valid_labels(record_paths: List[Path]) -> bool:
    """
    Checks if an individual has any labels other than 'unknown'.
    """
    for record_path in record_paths:
        try:
            with h5py.File(record_path, 'r') as f:
                labels_raw = f['label'][:]
                # Decode if they are byte strings
                if labels_raw.dtype.kind == 'S':
                    labels = np.char.decode(labels_raw, 'utf-8')
                else:
                    labels = labels_raw
                
                if np.any(labels != b'unknown'):
                    return True
        except Exception:
            # Ignore corrupted files or other read errors
            continue
    return False