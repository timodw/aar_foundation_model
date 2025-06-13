import h5py
from pathlib import Path
import numpy as np

from typing import Dict
from numpy.typing import NDArray


def dataset_to_hdf5(processed_data_path: Path, individual_id: int, dataset_id: int,
                    dataset: Dict[str, NDArray],
                    sampling_rate: int, mounting_location: str, species: str):
    ds_path = processed_data_path / f"{species.lower()}_{individual_id}_ds_{dataset_id}.hdf5"
    with h5py.File(ds_path, 'w') as f:
        f.attrs['sr'] = sampling_rate
        f.attrs['loc'] = mounting_location
        f.attrs['species'] = species
        for key, data in dataset.items():
            ds = f.create_dataset(key, data.shape, dtype=data.dtype if data.dtype != np.dtypes.ObjectDType else h5py.string_dtype())
            ds[:] = data
    return ds_path