from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bhenet import BHENet, CLASS_CODES, H5BuildingDataset, ModelConfig, build_bhenet, read_h5_sample


class TextStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)

    def forward(self, tokens):
        return SimpleNamespace(logits=self.embedding(tokens))


class ImageStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=1)

    def forward(self, image):
        return self.projection(image)


def fixture_model():
    return BHENet(TextStub(), ImageStub(), ModelConfig(feature_dim=8, head_dim=4, grid_size=2))


def inputs():
    return (
        torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.randn(2, 3, 2, 2),
        *(torch.randn(2, 1, 2, 2) for _ in range(3)),
        *(torch.randn(2, 3, 1) for _ in range(3)), torch.randn(2, 1, 2, 2),
        {"road": torch.randint(0, 2, (2, 1, 2, 2)).float()},
    )


def test_composed_forward_backward():
    model = fixture_model()
    output = model(*inputs())
    assert output.shape == (2, 29)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_strict_state_roundtrip():
    model = fixture_model().eval()
    clone = fixture_model().eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    values = inputs()
    torch.testing.assert_close(model(*values), clone(*values), atol=0, rtol=0)
    keys = model.state_dict().keys()
    assert any(k.startswith("language_model.") for k in keys)
    assert any(k.startswith("image_model_encoder.") for k in keys)
    assert any(k.startswith("fusion_model.") for k in keys)
    assert not any(k.startswith("image_model.") for k in keys)


@pytest.mark.parametrize("name,value", [("grid_size", 0), ("feature_dim", True), ("num_classes", 2.5)])
def test_bad_config(name, value):
    with pytest.raises(ValueError):
        ModelConfig(**{name: value})


def test_head_divisibility():
    with pytest.raises(ValueError):
        ModelConfig(feature_dim=7, head_dim=2)


def test_bad_token_dtype():
    values = list(inputs())
    values[0] = values[0].float()
    with pytest.raises(TypeError, match="integer"):
        fixture_model()(*values)


def test_bad_position_shape():
    values = list(inputs())
    values[5] = torch.randn(2, 4, 1)
    with pytest.raises(ValueError, match="shape"):
        fixture_model()(*values)


def test_prepared_float_input_required():
    values = list(inputs())
    values[1] = values[1].byte()
    with pytest.raises(TypeError, match="floating"):
        fixture_model()(*values)


def test_output_order():
    assert len(CLASS_CODES) == len(set(CLASS_CODES)) == 29
    assert CLASS_CODES[0] == "A" and CLASS_CODES[21] == "R" and CLASS_CODES[28] == "TC"


def test_builder_rejects_incompatible_dimensions_before_loading():
    with pytest.raises(ValueError, match="MobileSAM"):
        build_bhenet("absent", "absent", config=ModelConfig(feature_dim=8, head_dim=4))


def test_different_backbone_dtypes_are_bridged():
    model = fixture_model()
    model.language_model.double()
    model.image_model_encoder.double()
    assert model(*inputs()).dtype == torch.float32


@pytest.fixture
def synthetic_h5(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    path = tmp_path / "synthetic_building.h5"
    with h5py.File(path, "w") as file:
        file["BF1"] = "R"
        file["BF2"] = "None"
        file["index"] = 17
        file.create_dataset("pois_infor", data=["示例商店", "Example school"], dtype=h5py.string_dtype("utf-8"))
        file["build_center"] = np.array([0.4, 0.6], dtype=np.float64)
        file["pois_cor_x"] = np.array([0.2, 0.7], dtype=np.float64)
        file["pois_cor_y"] = np.array([0.3, 0.8], dtype=np.float64)
        file["x_mat"] = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
        file["y_mat"] = np.arange(16, dtype=np.float32).reshape(4, 4) / 15
        file["out_image"] = np.arange(3 * 5 * 6, dtype=np.uint8).reshape(3, 5, 6)
        file["out_label_0"] = np.array([[[0, 1, 2], [0, -1, 1]]], dtype=np.int16)
        file["out_road"] = np.array([[[0, 1, 0], [1, 0, 1]]], dtype=np.uint8)
    return path


def test_h5_stored_values_and_types(synthetic_h5):
    import numpy as np
    sample = read_h5_sample(synthetic_h5)
    assert sample["BF1"] == "R" and sample["BF2"] == "None"
    assert sample["index"] == 17 and isinstance(sample["index"], int)
    assert sample["pois_infor"] == ["示例商店", "Example school"]
    assert sample["pois_cor_x"] == [0.2, 0.7]
    assert sample["build_center"] == [0.4, 0.6]
    assert sample["x_mat"].shape == (4, 4) and sample["x_mat"].dtype == torch.float32
    assert sample["out_image"].dtype == np.uint8
    assert sample["out_image"].shape == (3, 5, 6)
    assert sample["out_label_0"].tolist() == [[[0, 1, 2], [0, -1, 1]]]


def test_h5_dataset_keeps_order_and_duplicates(synthetic_h5):
    dataset = H5BuildingDataset([synthetic_h5, synthetic_h5])
    assert len(dataset) == 2
    assert dataset[0]["filename"] == dataset[1]["filename"] == str(synthetic_h5)
    assert len(H5BuildingDataset([])) == 0
    with pytest.raises(TypeError, match="sequence"):
        H5BuildingDataset(str(synthetic_h5))


@pytest.mark.parametrize("field", ["BF1", "out_image", "x_mat"])
def test_h5_missing_field(synthetic_h5, field):
    import h5py
    with h5py.File(synthetic_h5, "r+") as file:
        del file[field]
    with pytest.raises(ValueError, match=field):
        read_h5_sample(synthetic_h5)


def test_h5_misaligned_poi(synthetic_h5):
    import h5py
    with h5py.File(synthetic_h5, "r+") as file:
        del file["pois_cor_x"]
        file["pois_cor_x"] = [0.2]
    with pytest.raises(ValueError, match="POI descriptions"):
        read_h5_sample(synthetic_h5)


def test_h5_rejects_wrong_raster_shape(synthetic_h5):
    import h5py
    import numpy as np
    with h5py.File(synthetic_h5, "r+") as file:
        del file["out_image"]
        file["out_image"] = np.zeros((5, 6, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="out_image"):
        read_h5_sample(synthetic_h5)


def test_h5_file_closed_and_unchanged(synthetic_h5):
    import hashlib
    import h5py
    digest = hashlib.sha256(synthetic_h5.read_bytes()).hexdigest()
    read_h5_sample(synthetic_h5)
    assert hashlib.sha256(synthetic_h5.read_bytes()).hexdigest() == digest
    with h5py.File(synthetic_h5, "r+") as file:
        file.attrs["reader_closed"] = True


def test_h5_dataloader_iteration(synthetic_h5):
    from torch.utils.data import DataLoader
    loader = DataLoader(H5BuildingDataset([synthetic_h5]), batch_size=None)
    sample = next(iter(loader))
    assert sample["index"] == 17
    assert isinstance(sample["out_image"], torch.Tensor)


def test_h5_missing_file(tmp_path):
    pytest.importorskip("h5py")
    with pytest.raises(FileNotFoundError):
        read_h5_sample(tmp_path / "missing.h5")
