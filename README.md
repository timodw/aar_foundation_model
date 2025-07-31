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
To pretrain the foundation model Numpy arrays containing the training and evaluation segments must be created from the HDF5 files. To do this the `data/data_preparation_pretraining.py` script can be used. This takes in several CLI parameters to configure the creation of the training and validation arrays as explained below:

|Parameter                |Default value|Description|
|-------------------------|-------------|-----------|
|`--data_root`            |             |Path of the root folder where the processed HDF5 datasets are stored.|
|`--output_folder`        |             |Path to the folder where to store the processed pretraining numpy arrays.|
|`--datasets`             |             |Names of the folders/datasets in `data_root` to be used for pretraining.|
|`--train_ratio`          |0.5          |Percentage of individual subjects used for training (Range [0.0, 1.0]).|
|`--max_dataset_imbalance`|4.0          |The maximum imbalance allowed between the smallest and largest datasets included (i.e if the smallest dataset contains 10 hours of data and the largest dataset contains 100 hours of data, with a maximum dataset imbalance of 4.0 only 10 hours * 4.0 = 40 hours of data will be included in the training data).|
|`--oversampling_factor`  |5.0          |The amount by which each dataset will be oversampled (i.e. with an oversampling factor of 5.0 a dataset containing 10 hours of data will be sampled in such a way that the total duration of all segments in the training array will amount to 50 hours of data).|
|`--segment duration`     |10.0         |The duration of each individual segment in the pretraining arrays, in number of seconds.|
|`--max_window_length`    |1000         |The maximum length of segments in number of accelerometer samples. Segments shorter than the maximum window length will be zero padded until they reach the maximum window length.|
|`--random_seed`          |578          |The random seed used to ensure reproducibility.|

An example of how to call this script is given below, this will use 2 datasets for pretraining (horses_2022 and cows_2023). The horses dataset contains data from 20 animals captured at 25Hz and a total of 23 hours of data, and the cows dataset contains data from 100 animals captured at 100 Hz with a total of 500 hours of data. We will use a train ratio of 50%, a maximum dataset imbalance of 5, an oversampling factor of 4. The segments for each dataset will be 20 seconds long with a maximum window length of 20 seconds * 100Hz = 2000 samples.

`python data/data_preparation_pretraining.py --data_root /data/aar/processed_data --output_folder /data/aar/training_snapshots/pretraining --datasets horses_2022 cows_2023 --train_ratio .50 --max_dataset_imbalance 5 --oversampling_factor 4 --segment_duration 20 --max_window_length 2000`

This will result in a training dataset containing data from 10 horses and 50 cows, the other animals will be used for the validation dataset. If we assume that both the training and validation datasets contain an equal amount of hours of data (23 / 2 = 11.5 hours for the horses, and 500 / 2 = 250 hours for the cows) then we can calculate the resulting amount of segments as follows. The smallest dataset will be oversampled by the oversampling factor, so 11.5 hours * 4 = 46 hours, the largest dataset will then be capped to 46 hours * 5 = 230 hours. So for the horses there will be 46 hours / 20 seconds = 8 280 segments extracted, and for the cows there will be 41 000 segments extracted, each of a 20 second duration. The segments will be extracted from the input data by generating random starting indices, so no sliding window approach is used. The resulting training and validation arrays will be stored in the folder indicated by the `--output_folder` argument as `X_train.npy` and `X_val.npy`. The dimensions of these arrays is `(n_segments, 3)`, as each different 3-axis accelerometer in every dataset will be processed seperately.

# Model training
## Model architecture
![Foundation model base architecture](assets/foundation_model_base.png)
The base architecture used in the foundation model is a Transformer Encoder only stack based on the BERT family of models, meaning there are only 2 free parameters to define the architecture of the Transformer: The embedding size (`d_embedding`) and the number of layers (`n_layers`). The number of attention heads (`n_heads`) and the dimensionality of the feedforward neural networks (`d_feedforward`) are derived from these 2 defining parameters as follows:
```
n_heads = d_embedding // 64
d_feedforward = d_embedding * 4
```
To reduce computational complexity patching is used before embedding the input. This means grouping together samples from multiple timesteps and calculating the embedding jointly on these grouped samples. As proposed by the PatchTST paper each axis of the accelerometer data will be passed separately through the Transformer, so the Transformer model will only handle 1D inputs. Our implementation makes use of a learned positional encoding as also proposed by PatchTST and used by the GPT family of models, this learned embedding is initialized using a uniform distribution ***U(-0.02, 0.02)***. For the embedding layer, the model currently supports 2 different methods of calculating the embeddings of the patches:
- **Linear**: A regular learned linear embedding, which is just a vector-matrix multiplication between the flattened patches and the embedding weight matrix.
- **Conv**: A convolutional embedding that makes use of a simple Convolutional Neural Network to calculate the embedding, the implementation details of this network can be found in `models.transformer.ConvEmbedding`.

The Transformer Encoder stack is implemented in `models.transformer.PatchedTransformerEncoderStack` and contains several parameters to configure the Transformer Encoder stack. A full list of the parameters is given below:
|Parameter                |Default value|Description|
|-------------------------|-------------|-----------|
|`n_patches`              |             |The length of the Transformers context window in number of patches|
|`patch_size`             |             |The size of each patch, the context input window thus contains ***n_patches \* patch_size*** time-series samples.|
|`n_modalities`           |             |The number of modalities in the time-series signal, for a 3-axis accelerometer this would be 3, one for each axis.
|`d_embedding`            |128          |The embedding size of the Transformer network.|
|`n_layers`               |4            |The number of layers of the Transformer Encoder stack.|
`transformer_dropout`     |0.1          |The dropout percentage used by the Transformer.|
`embedding_type`          |'linear'     |The type of embedding used by the stack, can be either `linear` or `conv`.|

## Self-supervised pretraining
The model used for this pretraining task is implemented in `models.transformer.SelfSupervisedBackbone`. This takes in an initialized (but not trained) `models.transformer.TransformerEncoderStack` and a self-supervised head. Currently this project only supports self-supervised pretraining through masking, so the only self-supervised head that is currently implemented is a reconstruction head (`models.transformer.ReconstructionHead`). In this method of pretraining a certain percentage of the input patches will be masked (set to zero in our implementation), a reconstruction head is then appended after the Transformer Encoder stack that will reconstruct the masked patches based on the information it can gather from the surrounding non-masked patches. A visual overview of the self-supervised backbone model with the reconstruction head is given below:

![Foundation model masked pre-training architecture](assets/foundation_model_pretraining.png)

The pretraining routine implemented in `training/pretrain.py` largely follows the standard self-supervised masked pretraining as is commonly used in foundation models. However there are some distinct implementation details:
- **Embedding type**: Our approach supports 2 different embedding types, either a regular linear embedding or a convolution based embedding.
- **Umasked loss weight**: Our approach also calculates the loss over the non-masked parts, the idea behind this being that it would be forced to look at the entire sequence and learn to reconstruct the entire sequence, not only the masked parts, hopefully learning more meaningful representations in this way. Early experiments showed an unmasked loss weight of 0.0001 to work best.
- **Multiple axes**: Our approach supports 2 methods of handling time-series data containing multiple axes. In the `multi` mode all axes are combined and passed jointly through the Transformer stack. In the `single` mode each axis is handled as an indepent sample and passed individuallay through the Transformer.

The script used to pretrain the model using the self-supervised masking task can be found at `training/pretrain.py` and contains several CLI parameters to configure the model that will be trained as well as the training itself:
|Parameter                |Default value|Description|
|-------------------------|-------------|-----------| 
|`--data_path`            |             |The path to the folder containing the `X_train.npy` and `X_val.npy` training and evaluation arrays to be used for pretraining.|
|`--output_dir`           |                  |The path to which the training logs and model weights will be saved, each individual training experiment will generate a distinct timestamped folder containing the logs and weights for this experiment.|
|`--batch_size`           |512               |The batch size used during training.|
|`--epochs`               |200               |Epochs used for pretraining, note that there is no early stopping implemented yet.|
|`--d_embedding`          |128               |Embedding size of the Transformer backbone model.|
|`--n_layers`             |4                 |Number of layers of the Transformer backbone model.|
|`--transformer_dropout`  |0.1               |Dropout for the Transformer backbone model.|
|`--masking_ratio`        |0.5               |The percentage of patches to be zero masked during pretraining.|
|`--patch_size`           |25                |The size of the patches used by the Transformer backbone.|
|`--warmup_steps`         |4000              |The number of steps used to warmup the Adam-based NoamOptimizer.|
|`--noam_factor`          |1.0               |The factor value used by the Adam-based NoamOptimizer.|
|`--input_mode`           |multi             |Select if you want to pass the 3-axis data as one sample to the Transformer (`multi`) or if you want each axis to be passed as an individual sample (`single`).|
|`--unmasked_loss_weight` |0.0               |The weight of the reconstruction of the unmasked patches of the input in the loss function.
|`--embedding_type`       |linear            |Which type of embedding to use, either linear or conv.|

## Fine-tuning for classification
