"""Backbone construction tests using small, local dependency doubles."""

import sys
import types

import pytest
import torch
from torch import nn

from bhenet.backbones import build_backbones


@pytest.fixture
def local_paths(tmp_path):
    text = tmp_path / "text"
    text.mkdir()
    (text / "config.json").write_text('{"hidden_size": 12}', encoding="utf-8")
    vision = tmp_path / "mobile_sam.pt"
    vision.write_bytes(b"mock checkpoint")
    return text, vision


@pytest.fixture
def fake_dependencies(monkeypatch):
    calls = {"lora": [], "text_load": [], "vision_load": []}

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=12)
            self.q_proj = nn.Linear(12, 12)
            self.v_proj = nn.Linear(12, 12)
            self.lm_head = nn.Linear(12, 20)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["text_load"].append((path, kwargs))
            return cls().to(dtype=kwargs["torch_dtype"])

    class Image(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(4, 12)

    class Wrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_model = base
            self.lora_A = nn.Parameter(torch.ones(1))
            self.lora_B = nn.Parameter(torch.ones(1))
            # Intentionally leave base parameters trainable: the factory must
            # impose its own documented freeze policy after wrapping.

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs
            calls["lora"].append(kwargs)

    def load_vision(checkpoint):
        calls["vision_load"].append(checkpoint)
        return types.SimpleNamespace(image_encoder=Image())

    transformers = types.ModuleType("transformers")
    transformers.LlamaForCausalLM = Text
    peft = types.ModuleType("peft")
    peft.LoraConfig = Config
    peft.get_peft_model = lambda model, config: Wrapper(model)
    mobile_sam = types.ModuleType("mobile_sam")
    mobile_sam.sam_model_registry = {"vit_t": load_vision}
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "mobile_sam", mobile_sam)
    return calls


def test_local_loading_and_explicit_adapter_configuration(local_paths, fake_dependencies):
    text, image = build_backbones(*local_paths)
    settings = fake_dependencies["text_load"][0][1]
    assert settings["local_files_only"] is True
    assert settings["trust_remote_code"] is False
    assert settings["torch_dtype"] == torch.float32
    text_lora, image_lora = fake_dependencies["lora"]
    assert text_lora["target_modules"] == ["q_proj", "v_proj"]
    assert text_lora["task_type"] == "TOKEN_CLS"
    assert image_lora["target_modules"] == ["qkv"]
    for config in (text_lora, image_lora):
        assert config["r"] == 8
        assert config["lora_alpha"] == 32
        assert config["lora_dropout"] == 0.05
        assert config["use_rslora"] is True
        assert config["bias"] == "none"
    assert text.base_model.lm_head.in_features == 12
    assert text.base_model.lm_head.out_features == 256
    assert not text.base_model.lm_head.weight.requires_grad
    for model in (text, image):
        for name, value in model.named_parameters():
            assert value.requires_grad == name.startswith("lora_")


def test_projection_training_is_an_explicit_option(local_paths, fake_dependencies):
    text, _ = build_backbones(*local_paths, train_text_projection=True)
    assert text.base_model.lm_head.weight.requires_grad
    assert not text.base_model.q_proj.weight.requires_grad


def test_missing_local_paths_are_rejected_before_optional_imports(tmp_path):
    with pytest.raises(FileNotFoundError, match="text model directory"):
        build_backbones(tmp_path / "missing", tmp_path / "missing.pt")


def test_missing_local_config_is_rejected(local_paths):
    (local_paths[0] / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="config.json"):
        build_backbones(*local_paths)


def test_missing_optional_dependency_has_a_clear_message(local_paths, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="optional text backbone requires"):
        build_backbones(*local_paths)


def test_missing_vision_checkpoint_is_rejected(local_paths):
    local_paths[1].unlink()
    with pytest.raises(FileNotFoundError, match="MobileSAM checkpoint"):
        build_backbones(*local_paths)


def test_invalid_projection_policy_is_rejected(local_paths):
    with pytest.raises(TypeError, match="must be a bool"):
        build_backbones(*local_paths, train_text_projection="false")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 128},
        {"feature_dim": True},
        {"lora_rank": 0},
        {"lora_rank": True},
        {"lora_alpha": float("nan")},
        {"lora_dropout": 1},
        {"text_dtype": torch.int64},
    ],
)
def test_invalid_settings_are_rejected(local_paths, kwargs):
    with pytest.raises(ValueError):
        build_backbones(*local_paths, **kwargs)


def test_missing_lora_targets_are_detected(local_paths, fake_dependencies, monkeypatch):
    monkeypatch.delattr(sys.modules["transformers"].LlamaForCausalLM, "from_pretrained")

    def load(*args, **kwargs):
        invalid = nn.Linear(12, 12)
        invalid.config = types.SimpleNamespace(hidden_size=12)
        return invalid

    monkeypatch.setattr(sys.modules["transformers"].LlamaForCausalLM, "from_pretrained", load, raising=False)
    with pytest.raises(ValueError, match="LoRA target modules"):
        build_backbones(*local_paths)
