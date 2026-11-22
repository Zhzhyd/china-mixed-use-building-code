"""Lightweight persistence and explicit sample-list checks."""

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from bhenet.runtime import (
    capture_rng_state, load_checkpoint, load_model_weights, load_run_config,
    read_manifest, restore_rng_state, sample_signature, save_checkpoint, seed_everything,
)


def test_rng_round_trip():
    seed_everything(2025)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_full_checkpoint_roundtrip(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler,
                    epoch=1, metadata={"sample": "fixed"})
    with torch.no_grad():
        model.weight.zero_()
    payload = load_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler,
                              expected_metadata={"sample": "fixed"}, restore_rng=True)
    assert payload["epoch"] == 1
    assert scheduler.last_epoch == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_checkpoint(path, model, expected_metadata={"sample": "changed"})
    assert not list(tmp_path.glob("*.tmp"))


def test_explicit_legacy_state_import(tmp_path):
    model = nn.Linear(2, 2)
    path = tmp_path / "legacy.pt"
    torch.save({"net": model.state_dict()}, path)
    other = nn.Linear(2, 2)
    load_model_weights(path, other, state_key="net")
    torch.testing.assert_close(model.weight, other.weight)
    with pytest.raises(ValueError, match="full_v1"):
        load_checkpoint(path, other)
    torch.save({"net": {"module." + k: v for k, v in model.state_dict().items()}}, path)
    with pytest.raises(RuntimeError):
        load_model_weights(path, other, state_key="net")


def test_checkpoint_rejects_class_order_and_trainable_mismatch(tmp_path):
    model = nn.Linear(2, 2)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, epoch=0, metadata={})
    model.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        load_checkpoint(path, model)
    payload = torch.load(path, weights_only=True)
    payload["class_codes"] = list(reversed(payload["class_codes"]))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="class order"):
        load_checkpoint(path, model)


def test_manifests_and_content_signatures(tmp_path):
    first, second = tmp_path / "a.h5", tmp_path / "b.h5"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps(["b.h5", "a.h5"]), encoding="utf-8")
    assert read_manifest(manifest) == [second, first]
    before = sample_signature([first, second])
    assert before != sample_signature([second, first])
    first.write_bytes(b"new")
    assert before != sample_signature([first, second])
    manifest.write_text('["a.h5", "./a.h5"]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(manifest)
    manifest.write_text('["missing.h5"]', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        read_manifest(manifest)


def test_config_relative_paths_and_unknown_keys(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"model": {"text_model_dir": "weights/text"},
                               "training": {"output_dir": "run"},
                               "data": {"validation_manifest": None}}), encoding="utf-8")
    config = load_run_config(path)
    assert config["model"]["text_model_dir"] == str(tmp_path / "weights/text")
    assert config["data"]["validation_manifest"] is None
    path.write_text('{"training": {"epohs": 10}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_run_config(path)


def test_runtime_builds_raster_schema_from_shared_config(monkeypatch):
    import sys
    from types import SimpleNamespace
    import bhenet.runtime as runtime
    seen = {}
    sentinel = nn.Linear(1, 1)

    def construct(*args, **kwargs):
        seen.update(kwargs)
        return sentinel

    class Tokenizer:
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True}
            return cls()

    monkeypatch.setattr(runtime, "build_bhenet", construct)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(LlamaTokenizer=Tokenizer))
    config = {"model": {"text_model_dir": "text", "vision_checkpoint": "vision", "device": "cpu"},
              "preprocessing": {"raster_names": ["road", "land_cover"], "raster_channels": [1, 3]}}
    previous_precision = torch.get_float32_matmul_precision()
    try:
        model, _ = runtime.build_runtime(config)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    assert model is sentinel
    assert seen["config"].raster_names == ("road", "land_cover")
    assert seen["config"].raster_channels == (1, 3)
