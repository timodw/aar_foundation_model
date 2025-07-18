# AAR Time Series Foundation Model
## Set-up
Run `pip install -r requirements.txt` in the root of this repository.

## Data preparation
### Dataset to HDF5 conversion
All datasets should be converted to the HDF5 format, for this the `dataset_to_hdf5` function from `data/utils.py` can be used. Every contiguous recording should be its own HDF5 file. Each different sensor will be turned into a separate `Dataset` in the HDF5 file.

The `data` folder already contains several example scripts for converting datasets into the required format (i.e. `data/process_horsing_around.py`). An example of the resulting file structure is given below:

```
dataset_name
|-- species_0_ds_0.hdf5
|-- species_0_ds_1.hdf5
|-- species_0_ds_2.hdf5
|-- species_1_ds_0.hdf5
`-- species_1_ds_1.hdf5
```

### Pretraining data


