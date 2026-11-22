"""Train, restore and predict with the composed raster-enabled classifier."""

import csv
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bhenet import BHENet, ModelConfig, PreprocessConfig, prepare_sample, read_h5_sample
from bhenet.predict import run_prediction
from bhenet.runtime import load_checkpoint, validate_input_contract
from bhenet.train import run_training


class Tokenizer:
    def __call__(self, descriptions, *, return_tensors, padding):
        ids = torch.tensor([[1, 3, 2] for _ in descriptions])
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)

    def forward(self, inputs):
        return SimpleNamespace(logits=self.embedding(inputs))


class Image(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(self.conv(inputs), 2)


def make_model(names=("road", "land_cover"), channels=(1, 2)):
    return BHENet(Text(), Image(), ModelConfig(feature_dim=8, head_dim=4,
                    grid_size=2, raster_names=names, raster_channels=channels))


def test_full_multiraster_training_checkpoint_and_inference(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    files = []
    for i in range(2):
        path = tmp_path / f"building_{i}.h5"
        with h5py.File(path, "w") as f:
            f["BF1"], f["BF2"], f["index"] = "R", "None", i
            f.create_dataset("pois_infor", data=["Example shop"], dtype=h5py.string_dtype("utf-8"))
            f["build_center"] = [0.5, 0.5]
            f["pois_cor_x"], f["pois_cor_y"] = [0.2], [0.6]
            f["x_mat"], f["y_mat"] = np.array([[0, 1], [0, 1]]), np.array([[0, 0], [1, 1]])
            f["out_image"] = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4) + i
            f["out_label_0"] = np.ones((1, 4, 4), dtype=np.uint8)
            f["out_road"] = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
            f["out_land_cover"] = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 8
        files.append(path)
    preparation = PreprocessConfig(image_size=4, grid_size=2, min_tokens=4,
                                   raster_names=("road", "land_cover"), raster_channels=(1, 2))
    torch.manual_seed(21)
    model = make_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = run_training(model, Tokenizer(), files, [], tmp_path / "run", epochs=1,
                          preprocess_config=preparation, learning_rate=1e-3)
    for branch in ("raster_road", "raster_land_cover"):
        assert any(branch in name and not torch.equal(value, before[name])
                   for name, value in model.named_parameters())
    restored = make_model()
    saved = load_checkpoint(result["last_checkpoint"], restored)
    assert saved["model_contract"]["config"]["raster_names"] == ("road", "land_cover")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    record = read_h5_sample(files[0], raster_names=preparation.raster_names,
                            raster_channels=preparation.raster_channels)
    batch = prepare_sample(record, Tokenizer(), preparation)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(*batch.inputs), restored(*batch.inputs), atol=0, rtol=0)
    csv_path = tmp_path / "predictions.csv"
    prediction = run_prediction(make_model(), Tokenizer(), files[::-1], csv_path,
                                checkpoint=result["last_checkpoint"],
                                preprocess_config=replace(preparation, max_tokens=4))
    assert prediction["samples"] == 2
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["building_id"] for row in rows] == ["1", "0"]
    assert all(sum(float(row[f"score_{c}"]) for c in saved["class_codes"]) == pytest.approx(1) for row in rows)
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(result["last_checkpoint"], make_model(("road",), (1,)))


def test_model_preprocessing_contract_checks_order_channels_and_size():
    preparation = PreprocessConfig(grid_size=2, raster_names=("road", "land_cover"), raster_channels=(1, 2))
    model = make_model()
    validate_input_contract(model, preparation)
    for changed in (replace(preparation, grid_size=3),
                    replace(preparation, raster_names=("land_cover", "road"), raster_channels=(2, 1)),
                    replace(preparation, raster_channels=(1, 3))):
        with pytest.raises(ValueError, match="disagree"):
            validate_input_contract(model, changed)
