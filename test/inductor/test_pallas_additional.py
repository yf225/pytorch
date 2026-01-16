"""
Tests for Pallas backend (2D canonical form approach).

These tests verify that the simplified Pallas backend correctly handles:
- Reduction operations
- Permute/transpose operations
- Expand operations
- Combinations of the above
"""
import math
import unittest
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.utils._pallas import has_cpu_pallas


@unittest.skipUnless(has_cpu_pallas(), "requires Pallas CPU backend")
class PallasTests(TestCase):
    """Test suite for Pallas backend."""

    DEVICE = "cpu"

    def _compile(self, fn):
        """Compile function with Pallas backend."""
        return torch.compile(fn, backend="inductor", options={"cpu_backend": "pallas"})

    def test_simple_add(self):
        """Test simple element-wise addition."""
        def fn(x, y):
            return x + y

        x = torch.randn(64, device=self.DEVICE)
        y = torch.randn(64, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result, expected)

    def test_simple_mul(self):
        """Test simple element-wise multiplication."""
        def fn(x, y):
            return x * y

        x = torch.randn(32, 32, device=self.DEVICE)
        y = torch.randn(32, 32, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result, expected)

    def test_sum_reduction(self):
        """Test sum reduction over entire tensor."""
        def fn(x):
            return x.sum()

        x = torch.randn(1024, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result, expected)

    def test_sum_reduction_2d(self):
        """Test sum reduction over last dimension of 2D tensor."""
        def fn(x):
            return x.sum(dim=-1)

        x = torch.randn(32, 64, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_max_reduction(self):
        """Test max reduction."""
        def fn(x):
            return x.max()

        x = torch.randn(1024, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result, expected)

    def test_permute_2d(self):
        """Test 2D transpose/permute."""
        def fn(x):
            return x.permute(1, 0) * 2.0

        x = torch.randn(32, 64, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_permute_3d(self):
        """Test 3D permute."""
        def fn(x):
            return x.permute(2, 0, 1) + 1.0

        x = torch.randn(8, 16, 32, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_expand_simple(self):
        """Test simple expand operation."""
        def fn(x):
            return x.unsqueeze(0).expand(4, 32) + 1.0

        x = torch.randn(32, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_expand_2d(self):
        """Test 2D expand with broadcasting."""
        def fn(x, y):
            # x: (1, 32), y: (16, 32)
            # expand x to (16, 32) then add
            return x.expand(16, 32) + y

        x = torch.randn(1, 32, device=self.DEVICE)
        y = torch.randn(16, 32, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_reduction_permute_expand(self):
        """
        Test combining reduction + permute + expand.

        This is the key test case that exercises all three operations together.
        """
        def fn(x):
            # x: (2, 8, 16)
            # 1. Permute: (2, 8, 16) → (8, 2, 16)
            x = x.permute(1, 0, 2)
            # 2. Expand: (8, 2, 16) → (8, 2, 4, 16) with new dim 2
            x = x.unsqueeze(2).expand(8, 2, 4, 16)
            # 3. Reduce: sum over last dim → (8, 2, 4)
            return x.sum(dim=-1)

        x = torch.randn(2, 8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_softmax_pattern(self):
        """Test softmax-like pattern (reduction + normalization)."""
        def fn(x):
            # Softmax: exp(x - max) / sum(exp(x - max))
            max_val = x.max(dim=-1, keepdim=True).values
            exp_x = torch.exp(x - max_val)
            sum_exp = exp_x.sum(dim=-1, keepdim=True)
            return exp_x / sum_exp

        x = torch.randn(16, 64, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_rmsnorm_pattern(self):
        """Test RMSNorm-like pattern (reduction + normalization)."""
        def fn(x, weight):
            # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
            variance = (x * x).mean(dim=-1, keepdim=True)
            x_normed = x / torch.sqrt(variance + 1e-6)
            return x_normed * weight

        x = torch.randn(16, 64, device=self.DEVICE)
        weight = torch.randn(64, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, weight)
        expected = fn(x, weight)

        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_generated_code_no_runtime_helpers(self):
        """Verify generated code doesn't use runtime helpers."""
        def fn(x):
            return x.sum(dim=-1)

        x = torch.randn(32, 64, device=self.DEVICE)

        compiled = self._compile(fn)
        _, (code,) = run_and_get_code(compiled, x)

        # Check that Pallas V2 specific code is generated
        self.assertIn("jax", code.lower())
        self.assertIn("jnp", code.lower())

        # Verify NO runtime helpers (key improvement of V2)
        self.assertNotIn("_pallas_partial_reduce", code)
        self.assertNotIn("_pallas_expand_for_broadcast", code)
        self.assertNotIn("_ensure_broadcast_compatible", code)

    def test_complex_permute_expand_reduce_4d(self):
        """
        Complex 4D test: permute + expand + reduction.

        x: (4, 8, 16) -> permute (2,0,1) -> (16, 4, 8)
        -> unsqueeze(1) -> (16, 1, 4, 8)
        -> expand -> (16, 3, 4, 8)
        -> sum(dim=-1) -> (16, 3, 4)
        """
        def fn(x):
            x = x.permute(2, 0, 1)  # (4, 8, 16) -> (16, 4, 8)
            x = x.unsqueeze(1).expand(16, 3, 4, 8)  # -> (16, 3, 4, 8)
            return x.sum(dim=-1)  # -> (16, 3, 4)

        x = torch.randn(4, 8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_permute_expand_reduce_middle_dim(self):
        """
        Test reduction on middle dimension after permute + expand.

        x: (8, 4, 16) -> permute(1,2,0) -> (4, 16, 8)
        -> unsqueeze(0) -> (1, 4, 16, 8)
        -> expand -> (2, 4, 16, 8)
        -> sum(dim=2) -> (2, 4, 8)
        """
        def fn(x):
            x = x.permute(1, 2, 0)  # (8, 4, 16) -> (4, 16, 8)
            x = x.unsqueeze(0).expand(2, 4, 16, 8)  # -> (2, 4, 16, 8)
            return x.sum(dim=2)  # reduce middle dim -> (2, 4, 8)

        x = torch.randn(8, 4, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_expand_multiple_dims(self):
        """
        Test expand on multiple dimensions simultaneously.

        x: (1, 8, 1) -> expand -> (4, 8, 16)
        -> sum(dim=-1) -> (4, 8)
        """
        def fn(x):
            x = x.expand(4, 8, 16)  # expand dims 0 and 2
            return x.sum(dim=-1)

        x = torch.randn(1, 8, 1, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_permute_then_broadcast_add(self):
        """
        Test permute followed by broadcast addition.

        x: (16, 32) -> permute -> (32, 16)
        y: (32, 1) -> broadcast add with x -> (32, 16)
        """
        def fn(x, y):
            x = x.permute(1, 0)  # (16, 32) -> (32, 16)
            return x + y  # y broadcasts from (32, 1) to (32, 16)

        x = torch.randn(16, 32, device=self.DEVICE)
        y = torch.randn(32, 1, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_layernorm_like_pattern(self):
        """
        Test LayerNorm-like pattern with permute.

        This tests a realistic pattern where we need to:
        1. Compute mean and variance over last dim
        2. Normalize
        3. Apply affine transform with weight/bias
        """
        def fn(x, weight, bias):
            # x: (batch, seq, hidden) = (4, 8, 32)
            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + 1e-6)
            return x_norm * weight + bias

        x = torch.randn(4, 8, 32, device=self.DEVICE)
        weight = torch.randn(32, device=self.DEVICE)
        bias = torch.randn(32, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, weight, bias)
        expected = fn(x, weight, bias)

        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_attention_score_like_pattern(self):
        """
        Test attention-score-like pattern with permute and softmax.

        Simulates computing attention weights:
        1. Permute Q to align with K
        2. Compute scores
        3. Apply softmax
        """
        def fn(q, k):
            # q: (batch, heads, seq_q, dim) = (2, 4, 8, 16)
            # k: (batch, heads, seq_k, dim) = (2, 4, 12, 16)
            # Transpose k for matmul: (2, 4, 16, 12)
            k_t = k.permute(0, 1, 3, 2)
            # For simplicity, just do element-wise ops on k_t
            # and reduce
            scores = k_t.sum(dim=-1)  # -> (2, 4, 16)
            # softmax over last dim
            max_s = scores.max(dim=-1, keepdim=True).values
            exp_s = torch.exp(scores - max_s)
            return exp_s / exp_s.sum(dim=-1, keepdim=True)

        k = torch.randn(2, 4, 12, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(None, k)  # q not used in this simplified version
        expected = fn(None, k)

        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    # ==================== Complex View Op Tests ====================

    def test_multiple_unsqueeze(self):
        """
        Test multiple unsqueeze operations in sequence.

        x: (4, 8) -> unsqueeze(0) -> (1, 4, 8) -> unsqueeze(2) -> (1, 4, 1, 8)
        -> sum(dim=-1) -> (1, 4, 1)
        """
        def fn(x):
            x = x.unsqueeze(0)  # (4, 8) -> (1, 4, 8)
            x = x.unsqueeze(2)  # (1, 4, 8) -> (1, 4, 1, 8)
            return x.sum(dim=-1)  # -> (1, 4, 1)

        x = torch.randn(4, 8, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_unsqueeze_at_different_positions(self):
        """
        Test unsqueeze at beginning, middle, and end.

        Tests explicit unsqueeze tracking (not size-based guessing).
        Note: fn_end uses sum(dim=0) which requires non-last-dim reduction support.
        """
        def fn_begin(x):
            return x.unsqueeze(0).sum(dim=-1)  # (4, 8) -> (1, 4, 8) -> (1, 4)

        def fn_middle(x):
            return x.unsqueeze(1).sum(dim=-1)  # (4, 8) -> (4, 1, 8) -> (4, 1)

        x = torch.randn(4, 8, device=self.DEVICE)

        # Test begin and middle (last-dim reduction)
        for fn, name in [(fn_begin, "begin"), (fn_middle, "middle")]:
            compiled = self._compile(fn)
            result = compiled(x)
            expected = fn(x)
            self.assertEqual(result.shape, expected.shape, f"Shape mismatch for unsqueeze at {name}")
            self.assertEqual(result, expected, f"Value mismatch for unsqueeze at {name}")

    def test_unsqueeze_then_reduce_first_dim(self):
        """Test unsqueeze followed by reduction on first dimension."""
        def fn(x):
            return x.unsqueeze(-1).sum(dim=0)  # (4, 8) -> (4, 8, 1) -> (8, 1)

        x = torch.randn(4, 8, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_unsqueeze_with_size_one_dims(self):
        """
        Test unsqueeze when input already has size-1 dimensions.

        This is the tricky case that requires explicit tracking (not size matching).
        x: (1, 4) -> unsqueeze(0) -> (1, 1, 4) -> sum(dim=-1) -> (1, 1)
        """
        def fn(x):
            x = x.unsqueeze(0)  # (1, 4) -> (1, 1, 4)
            return x.sum(dim=-1)  # -> (1, 1)

        x = torch.randn(1, 4, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_permute_unsqueeze_permute(self):
        """
        Test permute -> unsqueeze -> permute chain.

        x: (4, 8, 16) -> permute(2,0,1) -> (16, 4, 8)
        -> unsqueeze(1) -> (16, 1, 4, 8)
        -> permute(0,2,1,3) -> (16, 4, 1, 8)
        -> sum(dim=-1) -> (16, 4, 1)
        """
        def fn(x):
            x = x.permute(2, 0, 1)  # (4, 8, 16) -> (16, 4, 8)
            x = x.unsqueeze(1)  # -> (16, 1, 4, 8)
            x = x.permute(0, 2, 1, 3)  # -> (16, 4, 1, 8)
            return x.sum(dim=-1)  # -> (16, 4, 1)

        x = torch.randn(4, 8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_squeeze_unsqueeze_roundtrip(self):
        """
        Test squeeze followed by unsqueeze.

        x: (4, 1, 8) -> squeeze(1) -> (4, 8) -> unsqueeze(0) -> (1, 4, 8)
        -> sum(dim=-1) -> (1, 4)
        """
        def fn(x):
            x = x.squeeze(1)  # (4, 1, 8) -> (4, 8)
            x = x.unsqueeze(0)  # -> (1, 4, 8)
            return x.sum(dim=-1)  # -> (1, 4)

        x = torch.randn(4, 1, 8, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    # ==================== Multi-Tensor Fusion Tests ====================

    def test_two_tensor_different_permutes(self):
        """
        Test fusion of two tensors with different permute patterns.

        x: (4, 8) -> permute(1, 0) -> (8, 4)
        y: (8, 4) (no permute)
        result: x + y -> (8, 4) -> sum(dim=-1) -> (8,)
        """
        def fn(x, y):
            x = x.permute(1, 0)  # (4, 8) -> (8, 4)
            return (x + y).sum(dim=-1)  # -> (8,)

        x = torch.randn(4, 8, device=self.DEVICE)
        y = torch.randn(8, 4, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_two_tensor_both_permuted(self):
        """
        Test fusion of two tensors where both are permuted differently.

        x: (2, 4, 8) -> permute(1, 2, 0) -> (4, 8, 2)
        y: (4, 2, 8) -> permute(0, 2, 1) -> (4, 8, 2)
        result: x * y -> (4, 8, 2) -> sum(dim=-1) -> (4, 8)
        """
        def fn(x, y):
            x = x.permute(1, 2, 0)  # (2, 4, 8) -> (4, 8, 2)
            y = y.permute(0, 2, 1)  # (4, 2, 8) -> (4, 8, 2)
            return (x * y).sum(dim=-1)  # -> (4, 8)

        x = torch.randn(2, 4, 8, device=self.DEVICE)
        y = torch.randn(4, 2, 8, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_three_tensor_mixed_views(self):
        """
        Test fusion of three tensors with mixed view operations.

        x: (4, 8) -> permute(1, 0) -> (8, 4)
        y: (1, 4) -> expand(8, 4) -> (8, 4)
        z: (8, 4) (no view)
        result: x + y + z -> (8, 4) -> sum(dim=0) -> (4,)

        Currently fails: multiple issues (multi-tensor fusion + dim=0 reduction).
        """
        def fn(x, y, z):
            x = x.permute(1, 0)  # (4, 8) -> (8, 4)
            y = y.expand(8, 4)  # (1, 4) -> (8, 4)
            return (x + y + z).sum(dim=0)  # -> (4,)

        x = torch.randn(4, 8, device=self.DEVICE)
        y = torch.randn(1, 4, device=self.DEVICE)
        z = torch.randn(8, 4, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y, z)
        expected = fn(x, y, z)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_fusion_permute_and_unsqueeze_expand(self):
        """
        Test fusion where one tensor is permuted and another is unsqueeze+expanded.

        x: (4, 8, 16) -> permute(2, 0, 1) -> (16, 4, 8)
        y: (4, 8) -> unsqueeze(0) -> (1, 4, 8) -> expand(16, 4, 8) -> (16, 4, 8)
        result: x + y -> (16, 4, 8) -> sum(dim=-1) -> (16, 4)
        """
        def fn(x, y):
            x = x.permute(2, 0, 1)  # (4, 8, 16) -> (16, 4, 8)
            y = y.unsqueeze(0).expand(16, 4, 8)  # (4, 8) -> (16, 4, 8)
            return (x + y).sum(dim=-1)  # -> (16, 4)

        x = torch.randn(4, 8, 16, device=self.DEVICE)
        y = torch.randn(4, 8, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_fusion_different_reduction_inputs(self):
        """
        Test fusion where tensors contribute to different parts of a computation.

        x: (8, 16) -> sum(dim=1) -> (8,)
        y: (8,) (used directly)
        result: x_reduced + y -> (8,)

        Tests reduction followed by pointwise op with broadcast input.
        """
        def fn(x, y):
            x_reduced = x.sum(dim=1)  # (8, 16) -> (8,)
            return x_reduced + y  # -> (8,)

        x = torch.randn(8, 16, device=self.DEVICE)
        y = torch.randn(8, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_fusion_reduction_with_reshaped_broadcast(self):
        """
        Test fusion where post-reduction input has multi-dim shape matching numel.

        x: (8, 16) -> sum(dim=1) -> (8,)
        y: (2, 4) (same numel as reduction output, different shape)
        result: x_reduced + y.flatten() -> (8,)
        """
        def fn(x, y):
            x_reduced = x.sum(dim=1)  # (8, 16) -> (8,)
            return x_reduced + y.flatten()  # (2, 4) -> (8,), then add

        x = torch.randn(8, 16, device=self.DEVICE)
        y = torch.randn(2, 4, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_fusion_broadcast_from_different_shapes(self):
        """
        Test fusion with broadcasting from different source shapes.

        x: (8, 1) -> broadcasts to (8, 16)
        y: (1, 16) -> broadcasts to (8, 16)
        z: (8, 16) -> no broadcast
        result: x + y + z -> (8, 16) -> sum() -> scalar
        """
        def fn(x, y, z):
            return (x + y + z).sum()

        x = torch.randn(8, 1, device=self.DEVICE)
        y = torch.randn(1, 16, device=self.DEVICE)
        z = torch.randn(8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y, z)
        expected = fn(x, y, z)

        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_fusion_contiguous_and_noncontiguous(self):
        """
        Test fusion mixing contiguous and non-contiguous (transposed) tensors.

        x: (4, 8) contiguous
        y: (8, 4).T -> (4, 8) non-contiguous view
        result: x + y -> (4, 8) -> sum(dim=-1) -> (4,)
        """
        def fn(x, y):
            y_t = y.T  # (8, 4) -> (4, 8) via transpose
            return (x + y_t).sum(dim=-1)

        x = torch.randn(4, 8, device=self.DEVICE)
        y = torch.randn(8, 4, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x, y)
        expected = fn(x, y)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    # ==================== Complex Chain Tests ====================

    def test_long_view_chain(self):
        """
        Test a long chain of view operations.

        x: (2, 4, 8, 16)
        -> permute(3, 1, 2, 0) -> (16, 4, 8, 2)
        -> sum(dim=2) -> (16, 4, 2)
        """
        def fn(x):
            x = x.permute(3, 1, 2, 0)  # (2, 4, 8, 16) -> (16, 4, 8, 2)
            return x.sum(dim=2)  # -> (16, 4, 2)

        x = torch.randn(2, 4, 8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_reduction_on_first_dim_after_permute(self):
        """
        Test reduction on first dimension after permute.

        x: (4, 8, 16) -> permute(1, 2, 0) -> (8, 16, 4)
        -> sum(dim=0) -> (16, 4)

        Currently fails: scheduler limitation with dim=0 reductions.
        """
        def fn(x):
            x = x.permute(1, 2, 0)  # (4, 8, 16) -> (8, 16, 4)
            return x.sum(dim=0)  # reduce first dim -> (16, 4)

        x = torch.randn(4, 8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_multiple_reductions(self):
        """
        Test multiple sequential reductions.

        x: (4, 8, 16) -> sum(dim=2) -> (4, 8) -> sum(dim=1) -> (4,)
        """
        def fn(x):
            x = x.sum(dim=2)  # (4, 8, 16) -> (4, 8)
            return x.sum(dim=1)  # -> (4,)

        x = torch.randn(4, 8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_keepdim_reduction(self):
        """
        Test reduction with keepdim=True.

        x: (4, 8, 16) -> sum(dim=1, keepdim=True) -> (4, 1, 16)
        -> permute(2, 0, 1) -> (16, 4, 1)
        """
        def fn(x):
            x = x.sum(dim=1, keepdim=True)  # (4, 8, 16) -> (4, 1, 16)
            return x.permute(2, 0, 1)  # -> (16, 4, 1)

        x = torch.randn(4, 8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    # ==================== Embedding and Transformer Tests ====================

    def test_embedding_rmsnorm(self):
        """Minimal repro for embedding + reduction broadcasting issue.

        Tests embedding lookup followed by RMSNorm which involves:
        - Embedding: (batch, seq) indices -> (batch, seq, dim) lookup
        - Reduction: mean over last dim with keepdim=True

        The bug occurs when the Pallas codegen generates flatten indexing
        for embedding with incompatible shapes:
        - jnp.arange(dim) has shape (dim,)
        - indirect var (token indices) has shape (batch, seq)
        These cannot broadcast: (64,) vs (2, 16)
        """

        class EmbeddingRMSNorm(torch.nn.Module):
            def __init__(self, vocab_size: int, dim: int, eps: float = 1e-5):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, dim)
                self.weight = torch.nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, tokens: torch.Tensor):
                # Embedding lookup: (batch, seq) -> (batch, seq, dim)
                h = self.embedding(tokens)
                # RMSNorm: reduction over last dim
                h_norm = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
                return h_norm * self.weight

        model = EmbeddingRMSNorm(vocab_size=256, dim=64)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        # Input: (batch=2, seq=16) token indices
        x = torch.randint(0, 256, (2, 16), device=self.DEVICE)

        with torch.no_grad():
            expected = model(x)

        compiled_model = self._compile(model)
        with torch.no_grad():
            result = compiled_model(x)

        self.assertEqual(result, expected)

    def test_embedding_rmsnorm_residual(self):
        """Minimal repro for embedding + rmsnorm + residual add issue.

        Tests embedding lookup followed by RMSNorm and residual connection.
        The issue occurs when an intermediate buffer has flattened shape (32, 64)
        while embedding output has multi-dim shape (2, 16, 64).

        This pattern appears in transformer models like Llama3 where:
        - Embedding output: (batch, seq, dim)
        - Intermediate from previous layer: might be (batch*seq, dim) flattened
        - These get added together in residual connections
        """

        class EmbeddingRMSNormResidual(torch.nn.Module):
            def __init__(self, vocab_size: int, dim: int, eps: float = 1e-5):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, dim)
                self.weight = torch.nn.Parameter(torch.ones(dim))
                self.eps = eps
                # Linear layer that produces intermediate (could flatten internally)
                self.proj = torch.nn.Linear(dim, dim, bias=False)

            def forward(self, tokens: torch.Tensor):
                # Embedding lookup: (batch, seq) -> (batch, seq, dim)
                h = self.embedding(tokens)
                # RMSNorm: reduction over last dim
                h_norm = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
                h_scaled = h_norm * self.weight
                # Residual add with projection (may trigger shape mismatch)
                return h + self.proj(h_scaled)

        model = EmbeddingRMSNormResidual(vocab_size=256, dim=64)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        # Input: (batch=2, seq=16) token indices
        x = torch.randint(0, 256, (2, 16), device=self.DEVICE)

        with torch.no_grad():
            expected = model(x)

        compiled_model = self._compile(model)
        with torch.no_grad():
            result = compiled_model(x)

        self.assertEqual(result, expected)

    def test_transformer_block_minimal(self):
        """Minimal repro for transformer block broadcasting issue.

        Pattern that fails:
            h = emb(tokens)
            h = h + proj(norm(h))   # Residual with linear inside
            h = final_norm(h)       # This triggers broadcasting error

        The issue is that the intermediate buffer from the linear projection
        has shape (batch*seq, dim) = (32, 64) while the embedding output has
        shape (batch, seq, dim) = (2, 16, 64). When the final norm tries to
        operate on the result, JAX sees incompatible shapes.

        This is the core pattern from Llama3 and other transformers where
        each transformer block has:
        - RMSNorm -> Attention/Linear -> Residual
        - RMSNorm -> FFN -> Residual
        - Final RMSNorm at the end of the model
        """

        class RMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float = 1e-5):
                super().__init__()
                self.eps = eps
                self.weight = torch.nn.Parameter(torch.ones(dim))

            def forward(self, x):
                output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                return output * self.weight

        class TransformerBlockMinimal(torch.nn.Module):
            def __init__(self, vocab_size: int = 256, dim: int = 64, eps: float = 1e-5):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, dim)
                self.norm1 = RMSNorm(dim, eps)
                self.proj = torch.nn.Linear(dim, dim, bias=False)
                self.norm2 = RMSNorm(dim, eps)

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                h = self.embedding(tokens)  # (batch, seq, dim)
                h = h + self.proj(self.norm1(h))  # Residual with linear
                return self.norm2(h)  # Final norm triggers error

        model = TransformerBlockMinimal()
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        x = torch.randint(0, 256, (2, 16), device=self.DEVICE)

        with torch.no_grad():
            expected = model(x)

        compiled_model = self._compile(model)
        with torch.no_grad():
            result = compiled_model(x)

        self.assertEqual(result, expected)

    def test_same_tensor_multiple_view_chains(self):
        """Test same tensor used in multiple different view chains.

        This tests a case where the same permuted tensor is used in two
        different paths with different transforms. This exercises the
        view_id extraction logic to ensure each path gets the correct
        transform chain.

        Regression test for upgrade mechanism conflict bug where:
        - Path 1: permute -> unsqueeze -> expand -> reduce
        - Path 2: permute -> reduce
        Both paths share the same permute, but need different view chains.
        """
        def fn(x):
            # x: (2, 8, 16)
            p = x.permute(1, 0, 2)  # (8, 2, 16)

            # Two different paths from p:
            # Path 1: expand then reduce
            y = p.unsqueeze(2).expand(8, 2, 4, 16).sum(dim=-1).sum(dim=-1)  # (8, 2)
            # Path 2: just reduce
            z = p.sum(dim=-1)  # (8, 2)

            return y + z  # both used, same shape

        x = torch.randn(2, 8, 16, device=self.DEVICE)

        compiled = self._compile(fn)
        result = compiled(x)
        expected = fn(x)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_llama3(self):
        """Test Llama 3 model architecture.

        This is adapted from the official Meta Llama 3 implementation:
        https://github.com/meta-llama/llama3/blob/main/llama/model.py

        Tests the Llama 3 architecture including:
        - RMSNorm (Root Mean Square Layer Normalization)
        - Rotary Position Embeddings (RoPE)
        - Grouped Query Attention (GQA)
        - SwiGLU Feed-Forward Network
        - Residual connections
        """
        # ============================================================
        # Llama 3 model from https://github.com/meta-llama/llama3
        # Adapted to use standard PyTorch (no FairScale dependencies)
        # ============================================================

        @dataclass
        class ModelArgs:
            dim: int = 64  # Small for testing (original: 4096)
            n_layers: int = 2  # Small for testing (original: 32)
            n_heads: int = 4  # Small for testing (original: 32)
            n_kv_heads: Optional[int] = 2  # For GQA (original: 8 for 70B)
            vocab_size: int = 256  # Small for testing
            multiple_of: int = 64  # Make SwiGLU hidden layer size multiple of this
            ffn_dim_multiplier: Optional[float] = None
            norm_eps: float = 1e-5
            rope_theta: float = 500000.0
            max_seq_len: int = 32

        class RMSNorm(torch.nn.Module):
            """Root Mean Square Layer Normalization."""

            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = torch.nn.Parameter(torch.ones(dim))

            def _norm(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

            def forward(self, x):
                output = self._norm(x.float()).type_as(x)
                return output * self.weight

        def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
            """Precompute the frequency tensor for rotary embeddings."""
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            t = torch.arange(end, device=freqs.device, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            return freqs_cis

        def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
            """Reshape frequency tensor for broadcasting with x."""
            ndim = x.ndim
            assert 0 <= 1 < ndim
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

        def apply_rotary_emb(
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cis: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Apply rotary embeddings to query and key tensors."""
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
            """Repeat key/value heads for grouped query attention."""
            bs, slen, n_kv_heads, head_dim = x.shape
            if n_rep == 1:
                return x
            return (
                x[:, :, :, None, :]
                .expand(bs, slen, n_kv_heads, n_rep, head_dim)
                .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
            )

        class Attention(torch.nn.Module):
            """Multi-head attention with Grouped Query Attention (GQA)."""

            def __init__(self, args: ModelArgs):
                super().__init__()
                self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
                self.n_heads = args.n_heads
                self.n_rep = self.n_heads // self.n_kv_heads
                self.head_dim = args.dim // args.n_heads

                self.wq = torch.nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
                self.wk = torch.nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
                self.wv = torch.nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
                self.wo = torch.nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

            def forward(
                self,
                x: torch.Tensor,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
            ):
                bsz, seqlen, _ = x.shape
                xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

                xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
                xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
                xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

                # Repeat k/v heads if n_kv_heads < n_heads (GQA)
                keys = repeat_kv(xk, self.n_rep)
                values = repeat_kv(xv, self.n_rep)

                xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)

                scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores + mask
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                output = torch.matmul(scores, values)
                output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
                return self.wo(output)

        class FeedForward(torch.nn.Module):
            """SwiGLU Feed-Forward Network."""

            def __init__(
                self,
                dim: int,
                hidden_dim: int,
                multiple_of: int,
                ffn_dim_multiplier: Optional[float],
            ):
                super().__init__()
                hidden_dim = int(2 * hidden_dim / 3)
                if ffn_dim_multiplier is not None:
                    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
                hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

                self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
                self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
                self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

            def forward(self, x):
                return self.w2(F.silu(self.w1(x)) * self.w3(x))

        class TransformerBlock(torch.nn.Module):
            """Single Transformer block with attention and feed-forward."""

            def __init__(self, layer_id: int, args: ModelArgs):
                super().__init__()
                self.n_heads = args.n_heads
                self.dim = args.dim
                self.head_dim = args.dim // args.n_heads
                self.attention = Attention(args)
                self.feed_forward = FeedForward(
                    dim=args.dim,
                    hidden_dim=4 * args.dim,
                    multiple_of=args.multiple_of,
                    ffn_dim_multiplier=args.ffn_dim_multiplier,
                )
                self.layer_id = layer_id
                self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
                self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

            def forward(
                self,
                x: torch.Tensor,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
            ):
                h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
                out = h + self.feed_forward(self.ffn_norm(h))
                return out

        class Transformer(torch.nn.Module):
            """Llama 3 Transformer model."""

            def __init__(self, params: ModelArgs):
                super().__init__()
                self.params = params
                self.vocab_size = params.vocab_size
                self.n_layers = params.n_layers

                self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim)
                self.layers = torch.nn.ModuleList()
                for layer_id in range(params.n_layers):
                    self.layers.append(TransformerBlock(layer_id, params))
                self.norm = RMSNorm(params.dim, eps=params.norm_eps)
                self.output = torch.nn.Linear(params.dim, params.vocab_size, bias=False)

                # Precompute rotary embeddings
                self.freqs_cis = precompute_freqs_cis(
                    params.dim // params.n_heads,
                    params.max_seq_len * 2,
                    params.rope_theta,
                )

            def forward(self, tokens: torch.Tensor):
                bsz, seqlen = tokens.shape
                h = self.tok_embeddings(tokens)
                self.freqs_cis = self.freqs_cis.to(h.device)
                freqs_cis = self.freqs_cis[:seqlen]

                # Causal mask
                mask = None
                if seqlen > 1:
                    mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                    mask = torch.triu(mask, diagonal=1)
                    mask = mask.type_as(h)

                for layer in self.layers:
                    h = layer(h, freqs_cis, mask)
                h = self.norm(h)
                output = self.output(h).float()
                return output

        # Small config for testing (Llama 3 70B would have much larger dims)
        args = ModelArgs(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,  # GQA: 2 KV heads shared among 4 query heads
            vocab_size=256,
            multiple_of=64,
            norm_eps=1e-5,
            rope_theta=500000.0,
            max_seq_len=32,
        )

        model = Transformer(args)
        model.eval()
        if self.DEVICE != "cpu":
            model = model.to(self.DEVICE)

        # Test input
        x = torch.randint(0, args.vocab_size, (2, 16), device=self.DEVICE)

        # Run eager
        with torch.no_grad():
            expected = model(x)

        # Run compiled
        compiled_model = self._compile(model)
        with torch.no_grad():
            result = compiled_model(x)

        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
