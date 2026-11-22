import hashlib
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bhenet.dataloader import H5BuildingDataset, read_h5_sample
from bhenet.preprocessing import PreprocessConfig, prepare_sample


def raw_sample():
    return {
        "BF1": "R", "BF2": "None", "index": 7, "filename": "synthetic.h5",
        "pois_infor": ["Example road-side shop"],
        "build_center": [0.5, 0.5], "pois_cor_x": [0.25], "pois_cor_y": [0.75],
        "x_mat": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "y_mat": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        "out_image": torch.zeros(3, 3, 4, dtype=torch.uint8),
        "out_label_0": torch.tensor([[[0, 1], [1, 0]]]),
        "rasters": {"road": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8)},
    }


def tokenizer(descriptions, *, return_tensors, padding):
    assert return_tensors == "pt" and padding is True
    return {
        "input_ids": torch.tensor([[1, 4, 5]] * len(descriptions)),
        "attention_mask": torch.ones(len(descriptions), 3, dtype=torch.long),
    }


def config(**kwargs):
    return PreprocessConfig(image_size=2, grid_size=2, min_tokens=4, **kwargs)


@pytest.fixture
def raster_h5(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    path = tmp_path / "building_with_rasters.h5"
    sample = raw_sample()
    with h5py.File(path, "w") as handle:
        for key in ("BF1", "BF2", "index"):
            handle[key] = sample[key]
        handle.create_dataset("pois_infor", data=sample["pois_infor"], dtype=h5py.string_dtype("utf-8"))
        for key in ("build_center", "pois_cor_x", "pois_cor_y", "x_mat", "y_mat", "out_image", "out_label_0"):
            handle[key] = np.asarray(sample[key])
        handle["out_road"] = np.asarray(sample["rasters"]["road"])
        handle["out_height"] = np.array([[[-10.25, 0.0, 12.5], [3.0, 4.0, 5.0]]], dtype=np.float64)
        handle["out_cover"] = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    return path


def test_default_road_input_retains_binary_values_separately_from_building():
    sample = raw_sample()
    result = prepare_sample(sample, tokenizer, config())
    assert len(result.inputs) == 10
    assert result.inputs[8].tolist() == [[[[1, 2], [2, 1]]]]
    assert list(result.inputs[9]) == ["road"]
    road = result.inputs[9]["road"]
    assert road.tolist() == [[[[0, 1], [1, 0]]]]
    assert road.dtype == torch.float32 and road.device.type == "cpu"
    road.zero_()
    assert sample["rasters"]["road"].tolist() == [[[0, 1], [1, 0]]]


def test_named_rasters_use_nearest_resize_and_preserve_configured_order():
    sample = raw_sample()
    sample["rasters"]["height"] = torch.tensor([[[-2.5, 8.5, 99.0], [4.25, 6.5, 10.0]]], dtype=torch.float64)
    sample["rasters"]["cover"] = torch.arange(24, dtype=torch.int16).reshape(2, 3, 4)
    settings = config(raster_names=("height", "road", "cover"), raster_channels=(1, 1, 2))
    prepared = prepare_sample(sample, tokenizer, settings).inputs[9]
    assert tuple(prepared) == settings.raster_names
    for name, channels in zip(settings.raster_names, settings.raster_channels):
        expected = F.interpolate(sample["rasters"][name].float()[None], size=(2, 2), mode="nearest")
        torch.testing.assert_close(prepared[name], expected, atol=0, rtol=0)
        assert prepared[name].shape == (1, channels, 2, 2)
    assert prepared["height"].min() == -2.5


@pytest.mark.parametrize("value", [-1.0, 2.0, 0.5, 1.000000000001, 1e-12])
def test_road_requires_exact_binary_occupancy_before_float32_rounding(value):
    sample = raw_sample()
    sample["rasters"]["road"] = torch.tensor([[[value]]], dtype=torch.float64)
    with pytest.raises(ValueError, match="binary occupancy"):
        prepare_sample(sample, tokenizer, config())


@pytest.mark.parametrize("name", ["road", "height"])
def test_named_rasters_reject_nonfinite_values(name):
    sample = raw_sample()
    sample["rasters"] = {name: torch.tensor([[[float("inf")]]])}
    with pytest.raises(ValueError, match="finite"):
        prepare_sample(sample, tokenizer, config(raster_names=(name,), raster_channels=(1,)))


@pytest.mark.parametrize("rasters", [{}, {"height": torch.zeros(1, 2, 2)}, {"road": torch.zeros(1, 2, 2), "extra": torch.zeros(1, 2, 2)}])
def test_named_raster_keys_must_match_configuration(rasters):
    sample = raw_sample()
    sample["rasters"] = rasters
    with pytest.raises(ValueError, match="match configured names"):
        prepare_sample(sample, tokenizer, config())


def test_missing_road_mapping_is_not_inferred_from_building_or_top_level_fields():
    sample = raw_sample()
    del sample["rasters"]
    sample["out_road"] = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="missing=.*road"):
        prepare_sample(sample, tokenizer, config())


@pytest.mark.parametrize("value", [torch.zeros(2, 2), torch.zeros(2, 2, 2), torch.zeros(1, 0, 2)])
def test_named_raster_channel_and_spatial_shapes_are_checked(value):
    sample = raw_sample()
    sample["rasters"]["road"] = value
    with pytest.raises(ValueError, match="nonempty shape"):
        prepare_sample(sample, tokenizer, config())


def test_explicit_empty_branch_configuration_retains_nine_inputs():
    sample = raw_sample()
    del sample["rasters"]
    settings = config(raster_names=(), raster_channels=())
    assert len(prepare_sample(sample, tokenizer, settings).inputs) == 9
    sample["rasters"] = {"road": torch.zeros(1, 2, 2)}
    with pytest.raises(ValueError, match="unexpected=.*road"):
        prepare_sample(sample, tokenizer, settings)


def test_preprocessing_raster_spec_roundtrips_and_has_no_branch_count_limit():
    names = tuple(f"source_{i}" for i in range(40))
    settings = config(raster_names=list(names), raster_channels=[1] * len(names))
    assert settings.raster_names == names
    assert PreprocessConfig(**asdict(settings)) == settings
    sample = raw_sample()
    sample["rasters"] = {name: torch.full((1, 1, 1), float(i)) for i, name in enumerate(names)}
    result = prepare_sample(sample, tokenizer, settings)
    assert tuple(result.inputs[9]) == names
    assert result.inputs[9]["source_39"].sum() == 4 * 39


def test_hdf5_reader_retains_named_raster_values_dtypes_and_file_contents(raster_h5):
    import h5py
    import numpy as np

    digest = hashlib.sha256(raster_h5.read_bytes()).hexdigest()
    sample = read_h5_sample(raster_h5, raster_names=("height", "road", "cover"), raster_channels=(1, 1, 2))
    assert list(sample["rasters"]) == ["height", "road", "cover"]
    assert sample["rasters"]["height"].dtype == np.float64
    assert sample["rasters"]["road"].dtype == np.uint8
    assert sample["rasters"]["cover"].dtype == np.int16
    assert sample["rasters"]["height"].tolist() == [[[-10.25, 0.0, 12.5], [3.0, 4.0, 5.0]]]
    sample["rasters"]["road"][:] = 0
    assert read_h5_sample(raster_h5)["rasters"]["road"].sum() == 2
    assert hashlib.sha256(raster_h5.read_bytes()).hexdigest() == digest
    with h5py.File(raster_h5, "r+") as handle:
        handle.attrs["reader_has_closed"] = True


def test_hdf5_default_requires_road_and_legacy_schema_is_explicit(raster_h5):
    import h5py

    with h5py.File(raster_h5, "r+") as handle:
        del handle["out_road"]
    with pytest.raises(ValueError, match="missing HDF5 datasets: out_road"):
        read_h5_sample(raster_h5)
    sample = read_h5_sample(raster_h5, raster_names=(), raster_channels=())
    assert sample["rasters"] == {}
    assert len(prepare_sample(sample, tokenizer, config(raster_names=(), raster_channels=())).inputs) == 9


@pytest.mark.parametrize("value,pattern", [
    ([[0, 1], [1, 0]], "nonempty shape"),
    ([[[0, 2]]], "binary occupancy"),
    ([[[0, 1.000000000001]]], "binary occupancy"),
    ([[[0, float("nan")]]], "finite"),
    ([[["a", "b"]]], "real numeric"),
])
def test_hdf5_rejects_invalid_road_data(raster_h5, value, pattern):
    import h5py
    import numpy as np

    with h5py.File(raster_h5, "r+") as handle:
        del handle["out_road"]
        data = np.asarray(value)
        handle["out_road"] = data.astype("S1") if data.dtype.kind == "U" else data
    with pytest.raises(ValueError, match=pattern):
        read_h5_sample(raster_h5)


def test_hdf5_additional_branch_must_exist_with_configured_channels(raster_h5):
    with pytest.raises(ValueError, match="out_missing"):
        read_h5_sample(raster_h5, raster_names=("missing",), raster_channels=(1,))
    with pytest.raises(ValueError, match="out_cover.*shape"):
        read_h5_sample(raster_h5, raster_names=("cover",), raster_channels=(3,))


def test_hdf5_dataset_propagates_branch_configuration_and_dataloader_conversion(raster_h5):
    dataset = H5BuildingDataset(
        [raster_h5, raster_h5], raster_names=("cover", "road"), raster_channels=(2, 1),
    )
    assert dataset.raster_names == ("cover", "road")
    assert dataset[0]["filename"] == dataset[1]["filename"]
    sample = next(iter(DataLoader(dataset, batch_size=None)))
    assert isinstance(sample["rasters"]["cover"], torch.Tensor)
    assert sample["rasters"]["cover"].shape == (2, 3, 4)
    result = prepare_sample(sample, tokenizer, config(raster_names=("cover", "road"), raster_channels=(2, 1)))
    assert tuple(result.inputs[9]) == ("cover", "road")
