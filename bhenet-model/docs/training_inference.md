# Training and inference

## Inputs and configuration

Both entry points use the per-building HDF5 schema in the README. List sample
files in JSON manifests, for example:

```json
["samples/building_001.h5", "samples/building_002.h5"]
```

Paths in a manifest are relative to that manifest. Paths in `config.example.json`
are relative to the configuration file. Supply separate training, validation and
prediction manifests. Validation is optional: set `validation_manifest` to `null`
to train for a fixed number of epochs and retain the final checkpoint. Keep the
test set separate from a validation set used for checkpoint selection.

The HDF5 fields `BF1` and `BF2` supply training labels, using the class order in
`CLASS_CODES`. Prediction ignores reference labels, so these two
text fields may contain empty strings in unlabelled prediction records.

## Shared input preparation

`prepare_sample` normalizes RGB values with means `(123.675, 116.28, 103.53)` and
standard deviations `(58.395, 57.12, 57.375)`, then resizes to 1024 × 1024 by
bilinear interpolation. Stored RGB values retain their original intensity scale
before this transformation. Building-raster positive values become 2, zeros
become 1, and negative values retain their value; nearest-neighbour interpolation
produces a 64 × 64 auxiliary grid.

The separate `out_road` field contains a binary road-occupancy mask, with 0 for
background and 1 for roads. It is checked before resizing and retains these
values through nearest-neighbour resizing to 64 × 64. Road fusion follows
building fusion. Additional named rasters are loaded from `out_<name>` fields
and processed in the configured `raster_names` order; their channels are listed
in `raster_channels`. All raster inputs share the image crop's geographic extent
and orientation. Their values and aligned position features enter independent
fusion blocks. Training and prediction use the same raster configuration.

POI descriptions retain their stored order. Each description's normalized
coordinates are repeated across its tokens; tokenizer padding is removed with
the corresponding attention mask. Distances use the stored normalized building
centre and coordinate grids. Empty POI inputs use the original placeholder text
at the building centre. Short sequences are right-padded to 512 with token ID 2
and zero position/distance values. These final padding tokens participate in
model attention. Training retains longer sequences; inference uses its explicit
`max_tokens` setting to truncate tokens and coordinates together. Set it to
`null` to retain the entire sequence.

## Training

```bash
python -m bhenet.train --config config.example.json
```

The trainer runs one building per optimizer update on one configured device.
It uses AdamW with learning rate 2 × 10⁻⁵ and weight decay 0.01, cross-entropy
loss, and `CosineAnnealingLR(T_max=400, eta_min=1e-5)`. The scheduler advances
after every optimizer update, including beyond its first 400 updates; this
retains the source implementation's per-update cosine schedule. The example
configuration requests 400 epochs with seed 2025. Change `epochs` explicitly
for the intended run. The input lists remain fixed and a seeded permutation
sets their order each epoch. Text-backbone precision is configured separately;
image and fusion modules use float32.

`samples.jsonl` records sample-level training and validation outputs.
`epochs.jsonl` records losses, overall 29-class accuracy, per-class sample/correct counts and macro recall.
`macro_recall` averages over all 29 classes, with absent classes contributing
zero; `represented_macro_recall` averages over classes present in the supplied
references. When a validation manifest is provided, the highest validation
`macro_recall` selects `best.pt`. `last.pt` always represents the latest committed
epoch. A fixed-epoch analysis can use `last.pt` directly.

## Checkpoints and continuation

Complete checkpoints save all model weights, class order, optimizer and
scheduler state, completed-epoch count, random states, input-file hashes and
preprocessing settings. The frozen replacement text projection is included.
Full-backbone checkpoints require substantial disk space; `last.pt` and
`best.pt` retain the current and selected model states.

```bash
python -m bhenet.train --config config.example.json --resume runs/bhenet/last.pt
```

The resume path is relative to the command's working directory. Continue in
the same run directory and keep the samples, preprocessing and optimizer
settings fixed. `epochs` is the total requested completed-epoch count and may
be increased. Recovery uses the committed `last.pt` state and restores the
random streams before the next epoch. A new run uses an empty output directory.
If a journal write is interrupted, recovery restores the committed epoch log
and retains the old journal as a recovery record. Sample logs retain attempt IDs.

To initialize from a compatible historical full classifier, set
`model.initial_weights` and `model.initial_state_key` (for example, `net`).
The explicitly selected tensor dictionary is loaded strictly. Parameter names
and shapes must match the architecture. This is weight initialization; full
optimizer/scheduler recovery uses the package's `last.pt` format.

## Inference

```bash
python -m bhenet.predict --config config.example.json
```

Set `inference.checkpoint` to the selected classifier, `predict_manifest` to the
ordered input list and `output_csv` to a new file. The package checks checkpoint
class order, architecture, preprocessing and tokenizer identity. For a compatible
historical full checkpoint, `checkpoint_state_key` explicitly selects its state
dictionary, such as `net`.

The CSV contains the source filename and building ID, predicted 29-class code,
`BF1p`/`BF2p`, the winning softmax score, all class scores, the original valid
token count and the final input length after truncation/padding. Softmax scores express model confidence
and are not calibrated probabilities. Every input is processed in manifest
order and the output file is published only after processing completes.

The packaged workflow uses explicit fixed manifests, single-device execution,
strict checkpoint loading and completed-epoch recovery. These make sample
selection, initialization and continuation controlled run settings.
