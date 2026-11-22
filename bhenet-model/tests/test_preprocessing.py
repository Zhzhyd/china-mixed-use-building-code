from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from bhenet.labels import CLASS_CODES
from bhenet.preprocessing import PreprocessConfig, prepare_sample


class FakeTokenizer:
    """Keep padding visible so tests can detect token/coordinate misalignment."""

    pad_token_id = 0

    def __init__(self, rows=None):
        self.rows = rows
        self.descriptions = None

    def __call__(self, descriptions, *, return_tensors, padding):
        assert return_tensors == "pt" and padding is True
        self.descriptions = descriptions
        rows = self.rows or [[101, 10 + i, 102] for i in range(len(descriptions))]
        width = max(map(len, rows))
        return SimpleNamespace(
            input_ids=torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            attention_mask=torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        )


def sample():
    return {
        "BF1": "R", "BF2": "C", "index": 17, "filename": "synthetic_building.h5",
        "pois_infor": ["Example shop", "Example school"],
        "build_center": [0.333333333333, 0.600000000001],
        "pois_cor_x": [0.12500000001, 0.70000000001],
        "pois_cor_y": [0.25000000001, 0.80000000001],
        "x_mat": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "y_mat": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        "out_image": torch.arange(45, dtype=torch.uint8).reshape(3, 3, 5),
        "out_label_0": torch.tensor([[[0, 1, -7], [2, 9, 0]]], dtype=torch.int16),
    }


def tiny_config(**kwargs):
    return PreprocessConfig(
        image_size=2, grid_size=2, min_tokens=8, raster_names=(), raster_channels=(), **kwargs,
    )


def historical_formula(record, tokenizer, config):
    """Numerical reference for the original collated training loop."""
    data = default_collate([record])
    pixel_mean = torch.tensor(config.pixel_mean).view(3, 1, 1)
    pixel_std = torch.tensor(config.pixel_std).view(3, 1, 1)
    image = torch.stack([(value.float() - pixel_mean) / pixel_std for value in data["out_image"]])
    image = F.interpolate(image, size=config.image_size, mode="bilinear")
    label = data["out_label_0"].float()
    label[label > 0] = 2
    label[label == 0] = 1
    label = F.interpolate(label, size=config.grid_size, mode="nearest")
    encoded = tokenizer([value[0] for value in data["pois_infor"]], return_tensors="pt", padding=True)
    words = encoded.input_ids.shape[1]
    poi_x = torch.tensor(data["pois_cor_x"]).unsqueeze(-1).repeat(1, words).reshape(-1)
    poi_y = torch.tensor(data["pois_cor_y"]).unsqueeze(-1).repeat(1, words).reshape(-1)
    mask = encoded.attention_mask.reshape(-1).bool()
    ids = encoded.input_ids.reshape(-1)[mask].unsqueeze(0)
    poi_x, poi_y = (value[mask].unsqueeze(0).unsqueeze(-1).float() for value in (poi_x, poi_y))
    x_mat, y_mat = data["x_mat"].unsqueeze(0), data["y_mat"].unsqueeze(0)
    center = data["build_center"]
    image_dist = ((x_mat - center[0]) ** 2 + (y_mat - center[1]) ** 2) ** 0.5
    poi_dist = ((poi_x - center[0]) ** 2 + (poi_y - center[1]) ** 2) ** 0.5
    padding = config.min_tokens - ids.shape[1]
    if padding > 0:
        ids = F.pad(ids, (0, padding), value=2)
        poi_x, poi_y, poi_dist = (
            F.pad(value, (0, 0, 0, padding), value=0) for value in (poi_x, poi_y, poi_dist)
        )
    return ids, image, x_mat.float(), y_mat.float(), image_dist.float(), poi_x, poi_y, poi_dist.float(), label


@pytest.mark.parametrize("grid_dtype", [torch.float32, torch.float64])
def test_numerical_match_to_historical_training_formula(grid_dtype):
    record = sample()
    record["x_mat"] = record["x_mat"].to(grid_dtype) + 0.123456789012
    record["y_mat"] = record["y_mat"].to(grid_dtype) + 0.234567890123
    tokenizer = FakeTokenizer([[101, 11, 102], [101, 21, 22, 102]])
    config = tiny_config()
    expected = historical_formula(record, tokenizer, config)
    prepared = prepare_sample(record, tokenizer, config)
    for actual, reference in zip(prepared.inputs, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        assert actual.device.type == "cpu"
    assert prepared.inputs[0].dtype == torch.long
    assert all(value.dtype == torch.float32 for value in prepared.inputs[1:])
    assert prepared.target.tolist() == [CLASS_CODES.index("RC")]
    assert prepared.target.dtype == torch.long
    assert prepared.identifier == 17 and prepared.filename == "synthetic_building.h5"
    assert prepared.token_count == 7


def test_padding_alignment_and_negative_auxiliary_values():
    prepared = prepare_sample(sample(), FakeTokenizer([[11, 12], [21, 22, 23]]), tiny_config())
    ids, _, _, _, _, poi_x, poi_y, poi_dist, label = prepared.inputs
    assert ids.tolist() == [[11, 12, 21, 22, 23, 2, 2, 2]]
    assert poi_x[0, :5, 0].tolist() == pytest.approx([0.125, 0.125, 0.7, 0.7, 0.7])
    assert poi_y[0, :5, 0].tolist() == pytest.approx([0.25, 0.25, 0.8, 0.8, 0.8])
    for value in (poi_x, poi_y, poi_dist):
        assert torch.count_nonzero(value[:, 5:]) == 0
    # With width 3 -> 2 nearest interpolation, source column 2 is not sampled.
    assert label.tolist() == [[[[1.0, 2.0], [2.0, 2.0]]]]
    record = sample()
    record["out_label_0"] = torch.tensor([[[-7.0, 0.0], [3.0, -2.0]]])
    assert prepare_sample(record, FakeTokenizer(), tiny_config()).inputs[-1].tolist() == [
        [[[-7.0, 1.0], [2.0, -2.0]]]
    ]


def test_empty_poi_uses_original_placeholder_at_building_center():
    record = sample()
    record.update(pois_infor=[], pois_cor_x=[], pois_cor_y=[], build_center=[0.25, 0.75])
    tokenizer = FakeTokenizer()
    result = prepare_sample(record, tokenizer, tiny_config())
    assert tokenizer.descriptions == ["名称：无有效名称，大类：无有效类别"]
    assert result.token_count == 3
    assert result.inputs[5][0, :3, 0].tolist() == [0.25] * 3
    assert result.inputs[6][0, :3, 0].tolist() == [0.75] * 3
    assert torch.count_nonzero(result.inputs[7]) == 0
    assert record["pois_infor"] == [] and record["pois_cor_x"] == []


def test_long_training_sequence_is_never_truncated():
    tokenizer = FakeTokenizer([list(range(1, 301)), list(range(301, 601))])
    config = PreprocessConfig(image_size=2, grid_size=2, raster_names=(), raster_channels=())
    result = prepare_sample(sample(), tokenizer, config)
    assert result.token_count == 600
    assert result.inputs[0].shape == (1, 600)
    assert result.inputs[0][0, -1] == 600
    assert all(value.shape == (1, 600, 1) for value in result.inputs[5:8])


def test_explicit_inference_cap_preserves_pretruncation_count():
    tokenizer = FakeTokenizer([[11, 12], [21, 22, 23]])
    config = PreprocessConfig(
        image_size=2, grid_size=2, min_tokens=4, max_tokens=4, raster_names=(), raster_channels=(),
    )
    result = prepare_sample(sample(), tokenizer, config, require_target=False)
    assert result.target is None and result.token_count == 5
    assert result.inputs[0].tolist() == [[11, 12, 21, 22]]
    assert all(value.shape == (1, 4, 1) for value in result.inputs[5:8])
    assert result.inputs[5][0, -1, 0].item() == pytest.approx(0.7)
    with pytest.raises(ValueError, match="Training does not truncate"):
        prepare_sample(sample(), tokenizer, config)


@pytest.mark.parametrize("bf1,bf2,code", [("E", "E", "E"), ("R", "None", "R"), ("R", "C", "RC")])
def test_class_composition(bf1, bf2, code):
    record = sample()
    record.update(BF1=bf1, BF2=bf2)
    assert prepare_sample(record, FakeTokenizer(), tiny_config()).target.tolist() == [CLASS_CODES.index(code)]


@pytest.mark.parametrize("bf1,bf2", [("C", "R"), ("None", "None"), ("R", None), ("unknown", "None")])
def test_bad_training_labels_and_inference_ignoring_labels(bf1, bf2):
    record = sample()
    record.update(BF1=bf1, BF2=bf2)
    with pytest.raises(ValueError, match="class|BF1"):
        prepare_sample(record, FakeTokenizer(), tiny_config())
    assert prepare_sample(record, FakeTokenizer(), tiny_config(), require_target=False).target is None


def test_inference_does_not_require_label_fields():
    record = sample()
    del record["BF1"], record["BF2"]
    assert prepare_sample(record, FakeTokenizer(), tiny_config(), require_target=False).target is None


def test_source_arrays_and_tensors_are_unchanged_and_unaliased():
    np = pytest.importorskip("numpy")
    record = sample()
    record["out_image"] = record["out_image"].numpy()
    record["out_label_0"] = record["out_label_0"].numpy()
    before = deepcopy(record)
    result = prepare_sample(record, FakeTokenizer(), tiny_config())
    for value in result.inputs:
        value.zero_()
    for name, original in before.items():
        if isinstance(original, torch.Tensor):
            torch.testing.assert_close(record[name], original, atol=0, rtol=0)
        elif isinstance(original, np.ndarray):
            np.testing.assert_array_equal(record[name], original)
        else:
            assert record[name] == original


@pytest.mark.parametrize("name,value", [
    ("pois_cor_x", [0.2]), ("pois_cor_y", [[0.2, 0.3]]),
    ("build_center", [0.2]), ("x_mat", torch.zeros(4, 4)),
    ("y_mat", torch.zeros(1, 2, 2)), ("out_image", torch.zeros(4, 3, 3)),
    ("out_label_0", torch.zeros(2, 2)), ("pois_infor", "one string"),
    ("pois_infor", [None, "valid"]),
])
def test_malformed_samples_report_the_field(name, value):
    record = sample()
    record[name] = value
    with pytest.raises(ValueError, match=name):
        prepare_sample(record, FakeTokenizer(), tiny_config())


@pytest.mark.parametrize("name", ["out_image", "out_label_0", "x_mat", "y_mat", "build_center", "pois_cor_x", "pois_cor_y"])
def test_nonfinite_inputs_are_rejected(name):
    record = sample()
    value = torch.as_tensor(record[name]).float().clone()
    value.reshape(-1)[0] = float("nan")
    record[name] = value
    with pytest.raises(ValueError, match=f"{name}.*finite"):
        prepare_sample(record, FakeTokenizer(), tiny_config())


@pytest.mark.parametrize("encoded,pattern", [
    ({"input_ids": torch.ones(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2)}, "number of POI"),
    ({"input_ids": torch.ones(2, 2), "attention_mask": torch.ones(2, 2)}, "integer token"),
    ({"input_ids": torch.ones(2, 2, dtype=torch.long), "attention_mask": torch.ones(2, 3)}, "match input_ids"),
    ({"input_ids": torch.ones(2, 2, dtype=torch.long), "attention_mask": torch.full((2, 2), float("nan"))}, "only 0 or 1"),
    ({"input_ids": torch.ones(2, 2, dtype=torch.long), "attention_mask": torch.zeros(2, 2)}, "active token"),
    ({"input_ids": torch.ones(2, 2, dtype=torch.long)}, "return input_ids"),
])
def test_invalid_tokenizer_output_is_rejected(encoded, pattern):
    with pytest.raises(ValueError, match=pattern):
        prepare_sample(sample(), lambda *args, **kwargs: encoded, tiny_config())


@pytest.mark.parametrize("kwargs", [
    {"image_size": 0}, {"grid_size": True}, {"min_tokens": 0},
    {"max_tokens": 511}, {"max_tokens": 512.0}, {"padding_token_id": -1},
    {"pixel_mean": (1, 2)}, {"pixel_std": (1, 0, 1)},
    {"pixel_mean": (1, float("inf"), 3)},
])
def test_invalid_preprocess_configuration(kwargs):
    with pytest.raises(ValueError):
        PreprocessConfig(**kwargs)


def test_default_and_serialized_config_are_frozen():
    config = PreprocessConfig()
    assert (config.image_size, config.grid_size, config.min_tokens, config.max_tokens, config.padding_token_id) == (
        1024, 64, 512, None, 2,
    )
    assert PreprocessConfig(**asdict(config)) == config
    assert PreprocessConfig(pixel_mean=list(config.pixel_mean)) == config
    assert config.raster_names == ("road",) and config.raster_channels == (1,)
    with pytest.raises(FrozenInstanceError):
        config.min_tokens = 1
