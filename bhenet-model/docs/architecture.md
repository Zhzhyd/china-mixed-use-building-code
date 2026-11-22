# Architecture and tensor contract

The default configuration is `feature_dim=256`, `head_dim=32`, `grid_size=64`,
`aux_channels=1`, `num_classes=29`, `raster_names=("road",)` and
`raster_channels=(1,)`. Smaller fusion dimensions support
lightweight tests. The local MobileSAM builder uses 256 channels and a
64-by-64 feature grid.

## Inputs to BHENet.forward

| Argument | Shape | Meaning |
| --- | --- | --- |
| `text` | B × L | Integer token IDs |
| `image` | B × 3 × H × W | Already prepared floating RGB input |
| `x_mat`, `y_mat`, `xy_mat_dist` | B × 1 × G × G each | Image position and centre-distance features |
| `pois_cor_x`, `pois_cor_y`, `pois_cor_dist` | B × L × 1 each | Token-aligned POI position and centre-distance features |
| `label` | B × A × G × G | Auxiliary building raster, not the classification target |
| `rasters` | Mapping of names to B × C × G × G tensors | Additional raster modalities; default key `road`, C = 1 |

The research image tensor is 1024 × 1024 and uses the backbone's normalized
input scale. The HDF5 reader supplies the stored crop and position fields;
the model accepts their prepared tensor representation. Token-level POI
coordinates repeat each description's coordinates across its token sequence.
Distance features use the same normalized coordinates as the stored positions.
All supplied text tokens participate in attention, including supplied padding.

The text module returns `.logits` of shape B × L × D; this field holds projected
token features rather than final building classes. The image encoder returns
B × D × G × G. The fusion head produces B × K unnormalized class logits.

## Fusion

Each position triple is linearly mapped to D dimensions and RMS-normalized.
The building raster is flattened, linearly mapped to D dimensions without a
bias, and RMS-normalized. The default blocks are evaluated in this order:

1. Image features query image-position features.
2. Text features query text-position features.
3. Position-conditioned image features query position-conditioned text features.
4. The resulting image tokens query the building-raster features.
5. The resulting tokens query position-conditioned road-raster features.

Additional raster blocks follow in `raster_names` order. For each added raster
R, its C channels are projected to D channels and RMS-normalized; the resulting
tokens are added to the image-position features on the same grid before serving
as attention keys and values:

`K_R = V_R = RMSNorm_R(Linear_R(R)) + p_img`.

The attention block applies its own key/value projections to these inputs.
Each raster block has independent parameters. There is no fixed limit in code
on the number of registered raster blocks; memory and computation increase
with each block. An all-zero road mask represents no roads in the crop and is
still processed by its configured block. Setting both raster configuration
tuples to empty selects building-only raster fusion.

For each block, the query is RMS-normalized before its query projection. Keys
and values are projected from the second input. Attention scores are divided by
sqrt(D), and softmax is evaluated in float32. This denominator is retained from
the research implementation; it is not sqrt(the per-head dimension).
After the output projection, the query residual is added and RMS-normalized.
The feed-forward update is `W2(SiLU(W1(h)) * W3(h))`, added to `h`.

The final spatial projection maps G² tokens to one learned value per channel;
the classifier maps D features to K logits. This is not average pooling.

## Backbone adaptation

The optional builder explicitly uses RS-LoRA with rank 8, alpha 32 and dropout
0.05. Text targets are `q_proj` and `v_proj`; vision targets are `qkv`.
The text wrapper retains `TOKEN_CLS`. The vocabulary head is replaced by a
hidden-size-to-256 bias-free projection before adapter wrapping.

That projection is explicitly frozen by default. This avoids depending on
implicit PEFT freezing behaviour. `train_text_projection=True` enables training
of this projection.
These choices are explicit model-constructor settings.
