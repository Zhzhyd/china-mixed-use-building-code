"""Serial raster fusion, spatial alignment and explicit branch schemas."""

import io
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bhenet.config import ModelConfig, normalize_raster_spec
from bhenet.fusion import FusionHead, RasterFusionStage
from bhenet.model import BHENet


def inputs(batch=2, tokens=5, dim=8, grid=3):
    axis = torch.linspace(0, 1, grid)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    positions = (x, y, torch.sqrt(x.square() + y.square()))
    return (
        torch.randn(batch, tokens, dim), torch.randn(batch, dim, grid, grid),
        *(value.reshape(1, 1, grid, grid).repeat(batch, 1, 1, 1) for value in positions),
        *(torch.randn(batch, tokens, 1) for _ in range(3)),
        torch.randn(batch, 1, grid, grid),
    )


def make_head(**kwargs):
    torch.manual_seed(211)
    return FusionHead(feature_dim=8, head_dim=4, grid_size=3, **kwargs)


def test_default_road_is_required_and_logits_have_29_classes():
    head = make_head()
    tensors = inputs()
    assert head.raster_names == ("road",)
    with pytest.raises(ValueError, match="missing=.*road"):
        head(*tensors)
    output = head(*tensors, {"road": torch.ones(2, 1, 3, 3)})
    assert output.shape == (2, 29)
    assert torch.isfinite(output).all()


def test_road_follows_building_and_extra_stages_follow_configuration_order():
    head = make_head(raster_names=("road", "landuse"), raster_channels=(1, 2))
    tensors = inputs()
    events, outputs, queries = [], {}, {}

    def capture_building(module, args, output):
        events.append("building")
        outputs["building"] = output.detach().clone()

    head.attention_fusion_block_2.register_forward_hook(capture_building)
    for name in head.raster_names:
        stage = head.raster_branches[f"raster_{name}"]

        def capture(module, args, output, name=name):
            events.append(name)
            queries[name] = args[0].detach().clone()
            outputs[name] = output.detach().clone()

        stage.register_forward_hook(capture)
    # Mapping insertion order does not change the declared fusion order.
    output = head(*tensors, {"landuse": torch.rand(2, 2, 3, 3), "road": torch.rand(2, 1, 3, 3)})
    assert events == ["building", "road", "landuse"]
    torch.testing.assert_close(queries["road"], outputs["building"], rtol=0, atol=0)
    torch.testing.assert_close(queries["landuse"], outputs["road"], rtol=0, atol=0)
    expected = head.o2(head.o1(outputs["landuse"].transpose(1, 2)).reshape(2, 8))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert head.raster_branches["raster_road"].projection is not head.raster_branches["raster_landuse"].projection
    assert head.raster_branches["raster_road"].attention is not head.raster_branches["raster_landuse"].attention


def test_road_value_and_spatial_layout_both_change_output():
    head, tensors = make_head(), inputs()
    zero = torch.zeros(2, 1, 3, 3)
    first, second = zero.clone(), zero.clone()
    first[:, :, 0, 0] = 1
    second[:, :, 2, 2] = 1
    assert first.sum() == second.sum()
    empty_output = head(*tensors, {"road": zero})
    first_output = head(*tensors, {"road": first})
    second_output = head(*tensors, {"road": second})
    assert not torch.allclose(empty_output, first_output)
    assert not torch.allclose(first_output, second_output)


def test_every_serial_stage_and_each_raster_receive_finite_gradients():
    head = make_head(raster_names=("road", "landuse", "slope"), raster_channels=(1, 2, 1))
    tensors = inputs()
    rasters = {name: torch.randn(2, channels, 3, 3, requires_grad=True)
               for name, channels in zip(head.raster_names, head.raster_channels)}
    head(*tensors, rasters).square().mean().backward()
    for name, raster in rasters.items():
        assert raster.grad is not None, name
        assert torch.isfinite(raster.grad).all(), name
        assert raster.grad.abs().sum() > 0, name
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_raster_stage_uses_projected_raster_plus_aligned_position_features():
    stage = RasterFusionStage(feature_dim=8, head_dim=4, grid_size=3, channels=2)
    query, positions, raster = torch.randn(2, 9, 8), torch.randn(2, 9, 8), torch.randn(2, 2, 3, 3)
    tokens = raster.reshape(2, 2, 9).transpose(1, 2)
    expected = stage.attention(queryd=query, keyd=stage.norm(stage.projection(tokens)) + positions)
    torch.testing.assert_close(stage(query, raster, positions), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="position_features must have shape"):
        stage(query, raster, positions[:, :-1])


@pytest.mark.parametrize("rasters,error,match", [
    ({}, ValueError, "missing"),
    ({"road": torch.zeros(2, 1, 3, 3), "extra": torch.zeros(2, 1, 3, 3)}, ValueError, "extra"),
    ({"road": torch.zeros(2, 2, 3, 3)}, ValueError, "must have shape"),
    ({"road": torch.zeros(2, 1, 2, 3)}, ValueError, "must have shape"),
    ({"road": torch.zeros(1, 1, 3, 3)}, ValueError, "must have shape"),
    ({"road": torch.zeros(2, 1, 3, 3, dtype=torch.int64)}, TypeError, "floating-point"),
    ({"road": torch.full((2, 1, 3, 3), float("nan"))}, ValueError, "finite"),
    ({"road": torch.full((2, 1, 3, 3), float("inf"))}, ValueError, "finite"),
    ({"road": None}, TypeError, "torch.Tensor"),
    (torch.zeros(2, 1, 3, 3), TypeError, "mapping"),
])
def test_invalid_raster_mapping_is_rejected(rasters, error, match):
    with pytest.raises(error, match=match):
        make_head()(*inputs(), rasters)


@pytest.mark.parametrize("names,channels", [
    ("road", (1,)), (("road",), "1"), (("road", "road"), (1, 1)),
    (("road",), ()), (("road",), (0,)), (("road",), (True,)), (("road",), (1.0,)),
    (("Road",), (1,)), (("road.type",), (1,)), (("0road",), (1,)),
    (("building",), (1,)), (("label_0",), (1,)), (("image",), (1,)),
])
def test_invalid_raster_configuration_is_rejected(names, channels):
    with pytest.raises(ValueError):
        normalize_raster_spec(names, channels)
    with pytest.raises(ValueError):
        ModelConfig(raster_names=names, raster_channels=channels)


def test_schema_accepts_lists_arbitrary_branch_count_and_safe_module_names():
    names = ["road", "training", "forward"] + [f"layer_{index}" for index in range(9)]
    channels = [1] * len(names)
    config = ModelConfig(raster_names=names, raster_channels=channels)
    assert config.raster_names == tuple(names)
    assert config.raster_channels == tuple(channels)
    head = make_head(raster_names=names, raster_channels=channels)
    assert len(head.raster_branches) == 12
    assert list(head.raster_branches) == [f"raster_{name}" for name in names]


def test_empty_schema_supports_original_nine_tensor_interface():
    head = make_head(raster_names=(), raster_channels=())
    tensors = inputs()
    assert not head.raster_branches
    torch.testing.assert_close(head(*tensors), head(*tensors, {}), rtol=0, atol=0)
    with pytest.raises(ValueError, match="extra"):
        head(*tensors, {"road": torch.zeros(2, 1, 3, 3)})


def test_multi_raster_state_dictionary_roundtrip():
    schema = dict(raster_names=("road", "landuse"), raster_channels=(1, 2))
    head, other = make_head(**schema), make_head(**schema)
    memory = io.BytesIO()
    torch.save(head.state_dict(), memory)
    memory.seek(0)
    other.load_state_dict(torch.load(memory, weights_only=True), strict=True)
    tensors = inputs()
    rasters = {"road": torch.rand(2, 1, 3, 3), "landuse": torch.rand(2, 2, 3, 3)}
    torch.testing.assert_close(head(*tensors, rasters), other(*tensors, rasters), rtol=0, atol=0)
    assert "convert_lab.weight" in head.state_dict()
    assert "attention_fusion_block_2.wq.weight" in head.state_dict()
    assert "raster_branches.raster_road.projection.weight" in head.state_dict()


def test_bhenet_passes_rasters_to_configured_stages_with_dtype_conversion():
    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 8)

        def forward(self, tokens):
            return SimpleNamespace(logits=self.embedding(tokens))

    class Image(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Conv2d(3, 8, 1)

        def forward(self, image):
            return nn.functional.adaptive_avg_pool2d(self.projection(image), (3, 3))

    model = BHENet(Text(), Image(), ModelConfig(feature_dim=8, head_dim=4, grid_size=3)).double()
    raw = inputs()
    tensors = (torch.ones(2, 5, dtype=torch.long), torch.randn(2, 3, 6, 6), *raw[2:])
    road = torch.rand(2, 1, 3, 3, requires_grad=True)
    output = model(*tensors, {"road": road})
    assert output.shape == (2, 29) and output.dtype == torch.float64
    output.square().mean().backward()
    assert road.grad is not None and torch.isfinite(road.grad).all()
    with pytest.raises(ValueError, match="missing"):
        model(*tensors)
