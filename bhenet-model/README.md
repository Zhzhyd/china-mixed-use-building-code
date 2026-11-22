# BHENet

A multimodal building-function classifier combining POI text features,
remote-sensing image features, spatial positions, building rasters and road rasters.
The package provides model components, an HDF5 dataset interface, shared input
preparation, training, checkpoint recovery and per-building inference.

## Architecture

The model uses an Atom-7B-Chat text backbone and the MobileSAM
`vit_t` image encoder with RS-LoRA adapters. Text tokens and image features have
256 channels. Cross-attention combines image positions, text positions,
text/image features and the building raster, followed by a separate road-raster
fusion block. Additional raster blocks can be appended in configuration order.
Each added block has its own projection, normalization and attention parameters,
and combines raster features with aligned position features. A learned spatial
projection and a classification layer produce 29 unnormalized logits.

The ordered classifier labels are provided in `bhenet.CLASS_CODES`.

```text
src/bhenet/
  config.py       Architecture dimensions
  layers.py       RMS normalization and cross-attention
  fusion.py       Spatial and multimodal fusion head
  backbones.py    Optional, local-weight backbone construction
  model.py        Composition of the backbone and fusion modules
  labels.py       Ordered classifier output codes
  dataloader.py   Per-building HDF5 schema and dataset reader
  preprocessing.py  Shared image, text and spatial input preparation
  runtime.py      Configuration, sample manifests and checkpoints
  train.py        Training, validation and epoch-boundary recovery
  predict.py      Ordered per-building predictions and class scores
tests/            Model and sample-interface tests
docs/             Architecture, environment and workflow instructions
config.example.json  Training and inference configuration
```

## Installation

Python 3.10 or newer and PyTorch support the model architecture. The `data` extra
provides NumPy and h5py for sample loading; the `test` extra provides pytest.
From this directory:

```bash
python -m pip install -e ".[data,test]"
python -m pytest
```

`build_bhenet` additionally uses Transformers, PEFT, MobileSAM and existing local
backbone files. These dependencies are loaded when construction is requested.
See [environment.md](docs/environment.md) for environment information.

## HDF5 samples

Each HDF5 file represents one building. `dataloader.py` documents and reads the
following datasets, preserving their values and order:

| Dataset | Shape | Content |
| --- | --- | --- |
| `BF1`, `BF2` | Scalar text | Background-function and building-use labels |
| `index` | Scalar | Building identifier |
| `pois_infor` | P text entries | Ordered POI descriptions |
| `build_center` | 2 | Normalized building-centre coordinates |
| `pois_cor_x`, `pois_cor_y` | P each | Normalized POI coordinates aligned with descriptions |
| `x_mat`, `y_mat` | G × G | Normalized image-position grids; research G = 64 |
| `out_image` | 3 × H × W | RGB crop in channel-first order |
| `out_label_0` | 1 × Ha × Wa | Single-channel building auxiliary raster |
| `out_road` | 1 × Hr × Wr | Binary road occupancy: 0 = background, 1 = road |

```python
from bhenet import H5BuildingDataset

dataset = H5BuildingDataset(["samples/building_001.h5", "samples/building_002.h5"])
sample = dataset[0]
```

The filenames above illustrate the expected file organization. The reader
returns labels and POI text as strings, coordinate vectors as lists, position
grids as torch tensors, and crop rasters as NumPy arrays. File handles close after
each read, and supplied file order is retained. POI counts and crop sizes may
vary by building. Individual records can be iterated using
`DataLoader(dataset, batch_size=None)`.
PyTorch's default conversion changes NumPy arrays into tensors during that
DataLoader iteration; direct `dataset[index]` access returns the types above.
The reader places the road raster in `sample["rasters"]["road"]`. Road and other
auxiliary rasters must cover the same geographic extent and orientation as the
image and position grids. Their stored pixel resolutions may differ.

The stored-record interface and the model-tensor interface represent successive
stages. Model preparation normalizes/resizes the RGB crop, encodes the auxiliary
raster, tokenizes the POI descriptions and aligns their coordinates with tokens.
Position-to-centre distances are calculated from the stored normalized
coordinates. The resulting tensor shapes are specified in
[architecture.md](docs/architecture.md).

The default `raster_names=["road"]` and `raster_channels=[1]` append road fusion
after building fusion. For example, adding a three-channel raster uses
`raster_names=["road", "land_cover"]` and `raster_channels=[1, 3]`, with HDF5
datasets `out_road` and `out_land_cover`. The same settings are used by the
reader, preprocessing and model. Extra rasters retain their finite numeric
values through nearest-neighbour resizing. The road raster is checked for
binary 0/1 values. Configured raster inputs are required.

## Model API

`BHENet(language_model, image_model_encoder, config)` composes explicitly supplied
PyTorch modules. `FusionHead` operates on encoded features.
`build_bhenet(text_model_dir, vision_checkpoint, ...)` constructs the research
backbone family from local files and initializes adapter, projection and fusion
parameters. A trained classifier is restored by loading its matching complete
state dictionary using `load_state_dict(state_dict, strict=True)`.

CPU is the default construction device. Device placement is explicit, with
feature transfers at backbone/fusion boundaries. Shape and type checks give
specific errors for incompatible inputs.

## Training and inference

Configure local backbone paths, HDF5 sample manifests and output locations in
`config.example.json`. Each manifest is a JSON list of sample paths, resolved
relative to the manifest itself. Configuration paths are resolved relative to
the configuration file.
Raster names and channel counts in the `preprocessing` section also configure
the ordered model blocks and are recorded in each checkpoint.

```bash
python -m bhenet.train --config config.example.json
python -m bhenet.train --config config.example.json --resume runs/bhenet/last.pt
python -m bhenet.predict --config config.example.json
```

Training uses AdamW, cross-entropy loss and a per-update cosine learning-rate
schedule. Complete checkpoints retain model weights, optimizer state, the
scheduler, random states, input signatures and preprocessing settings. Validation
reports overall accuracy, macro recall and per-class counts. Inference exports the building
identifier, predicted class, background/use components and all 29 softmax scores.

See [training and inference](docs/training_inference.md) for configuration,
checkpoint selection, recovery and output definitions. Tests cover input
preparation, model composition, training steps, recovery and ordered predictions.

## Licences

This project is licensed under the [MIT License](LICENSE). Third-party components
and backbone weights retain their respective licences and use terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
