"""Small CPU-only tests for the model's fusion components."""

from __future__ import annotations

import io
import math
import unittest

import torch
from torch.nn import functional as F

from bhenet.fusion import FusionHead
from bhenet.layers import CrossAttentionBlock, RMSNorm


def make_inputs(batch: int = 2, tokens: int = 5, dim: int = 8, grid: int = 3):
    return (
        torch.randn(batch, tokens, dim),
        torch.randn(batch, dim, grid, grid),
        *(torch.randn(batch, 1, grid, grid) for _ in range(3)),
        *(torch.randn(batch, tokens, 1) for _ in range(3)),
        torch.randn(batch, 1, grid, grid),
    )


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2025)
        self.model = FusionHead(feature_dim=8, head_dim=4, grid_size=3, num_classes=7,
                                raster_names=(), raster_channels=())
        self.inputs = make_inputs()

    def test_shape_and_backward(self):
        inputs = tuple(tensor.requires_grad_() for tensor in self.inputs)
        output = self.model(*inputs)
        self.assertEqual(tuple(output.shape), (2, 7))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for tensor in inputs:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_noncontiguous_inputs(self):
        inputs = tuple(
            tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
            for tensor in self.inputs
        )
        self.assertFalse(inputs[0].is_contiguous())
        self.assertFalse(inputs[1].is_contiguous())
        torch.testing.assert_close(self.model(*inputs), self.model(*self.inputs), rtol=0, atol=0)

    def test_state_dictionary_roundtrip(self):
        memory = io.BytesIO()
        torch.save(self.model.state_dict(), memory)
        memory.seek(0)
        other = FusionHead(feature_dim=8, head_dim=4, grid_size=3, num_classes=7,
                           raster_names=(), raster_channels=())
        other.load_state_dict(torch.load(memory, weights_only=True))
        torch.testing.assert_close(self.model(*self.inputs), other(*self.inputs), rtol=0, atol=0)
        keys = set(other.state_dict())
        self.assertEqual(len(keys), 44)
        for key in (
            "p_convert_txt.weight", "p_convert_img.weight", "convert_lab.weight",
            "txt_posi_fusion.attention_norm.weight", "img_posi_fusion.wq.weight",
            "attention_fusion_block_1.w3.weight", "attention_fusion_block_2.w2.weight",
            "o1.weight", "o2.weight",
        ):
            self.assertIn(key, keys)

    def test_attention_matches_explicit_equation(self):
        block = CrossAttentionBlock(feature_dim=8, head_dim=4)
        query, key = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
        actual, scores = block(query, key, ifreturn_scores=True)
        normalized_query = block.attention_norm(query)
        q = block.wq(normalized_query).reshape(2, 3, 2, 4).transpose(1, 2)
        k = block.wk(key).reshape(2, 5, 2, 4).transpose(1, 2)
        v = block.wv(key).reshape(2, 5, 2, 4).transpose(1, 2)
        expected_scores = F.softmax((q @ k.transpose(2, 3) / math.sqrt(8)).float(), dim=-1).type_as(q)
        expected = block.wo((expected_scores @ v).transpose(1, 2).reshape(2, 3, 8))
        expected = block.ffn_norm(query + expected)
        expected = expected + block.w2(F.silu(block.w1(expected)) * block.w3(expected))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
        torch.testing.assert_close(scores.sum(dim=-1), torch.ones(2, 2, 3))

    def test_image_query_and_sequence_query_agree(self):
        block = CrossAttentionBlock(feature_dim=8, head_dim=4)
        image, key = torch.randn(2, 8, 3, 3), torch.randn(2, 5, 8)
        sequence = image.reshape(2, 8, 9).transpose(1, 2)
        torch.testing.assert_close(block(image, key), block(sequence, key), rtol=0, atol=0)

    def test_rmsnorm_equation(self):
        norm = RMSNorm(8)
        x = torch.randn(2, 5, 8)
        expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)

    def test_invalid_constructor_arguments(self):
        for kwargs in ({"feature_dim": 7}, {"grid_size": 0}, {"aux_channels": -1}, {"num_classes": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FusionHead(**kwargs)
        with self.assertRaises(ValueError):
            RMSNorm(8, eps=float("nan"))

    def test_invalid_shapes_and_types(self):
        changes = (
            (0, torch.randn(2, 0, 8), ValueError),
            (0, torch.randn(2, 5, 9), ValueError),
            (1, torch.randn(2, 8, 4, 4), ValueError),
            (2, torch.randn(2, 3, 3), ValueError),
            (5, torch.randn(2, 4, 1), ValueError),
            (8, torch.randn(2, 2, 3, 3), ValueError),
            (8, torch.ones(2, 1, 3, 3, dtype=torch.int64), TypeError),
            (8, self.inputs[8].double(), TypeError),
            (8, None, TypeError),
        )
        for index, value, error in changes:
            inputs = list(self.inputs)
            inputs[index] = value
            with self.subTest(index=index, value_type=type(value)), self.assertRaises(error):
                self.model(*inputs)

    def test_model_dtype_mismatch_is_explicit(self):
        with self.assertRaisesRegex(TypeError, "parameters use"):
            self.model(*(tensor.double() for tensor in self.inputs))

    def test_device_mismatch_is_explicit(self):
        inputs = list(self.inputs)
        inputs[8] = torch.empty(inputs[8].shape, device="meta")
        with self.assertRaisesRegex(ValueError, "expected"):
            self.model(*inputs)


if __name__ == "__main__":
    unittest.main()
