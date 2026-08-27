# Copyright © 2026 Apple Inc.

"""Pin the packed gated-delta kernel to the kernel it replaces.

The packed kernel must be bit-identical to the explicit-tree comparator by
construction, and on Apple GPUs also to the pre-existing simd_sum kernel.
Anything the packed path does not support must fall back to the original.
Uses random inputs -- no model download required.
"""

import unittest

import mlx.core as mx

import mlx_lm.models.gated_delta as gated_delta
from mlx_lm.models.gated_delta import (
    compute_g,
    gated_delta_kernel,
    gated_delta_kernel_packed,
    gated_delta_kernel_unpacked,
    gated_delta_kernel_xtree,
    gated_delta_ops,
)

# B, Hk, Hv, Dk, Dv, dtype
PACKED_CASES = [
    (1, 16, 32, 128, 128, mx.bfloat16),  # Qwen3.5/3.6 shape
    (2, 16, 32, 128, 128, mx.bfloat16),  # batched
    (1, 4, 8, 128, 128, mx.bfloat16),  # fewer heads
    (1, 8, 8, 128, 128, mx.bfloat16),  # Hv == Hk
    (1, 16, 32, 128, 128, mx.float16),
    (1, 16, 32, 128, 128, mx.float32),
    (1, 8, 16, 128, 64, mx.bfloat16),  # Dv != Dk
    (3, 2, 8, 128, 256, mx.bfloat16),  # larger Dv
]


def _make_inputs(B, T, Hk, Hv, Dk, Dv, dtype, seed=3):
    mx.random.seed(seed)

    def normed(shape, dim):
        x = mx.random.normal(shape)
        return (mx.fast.rms_norm(x, None, 1e-6) * dim**-0.5).astype(dtype)

    q = normed((B, T, Hk, Dk), Dk)
    k = normed((B, T, Hk, Dk), Dk)
    v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
    a = (mx.random.normal((B, T, Hv)) * 0.5).astype(dtype)
    b = mx.random.normal((B, T, Hv)).astype(dtype)
    A_log = mx.log(mx.random.uniform(low=0.5, high=16.0, shape=(Hv,)))
    dt_bias = mx.ones((Hv,))
    state = (mx.random.normal((B, Hv, Dv, Dk)) * 0.3).astype(mx.float32)
    mx.eval(q, k, v, a, b, A_log, dt_bias, state)
    return q, k, v, a, b, A_log, dt_bias, state


def _rel_l2(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-9)).item()


@unittest.skipUnless(
    mx.metal.is_available() and mx.default_device() == mx.gpu,
    "gated delta kernels are GPU only",
)
class TestGatedDeltaPacked(unittest.TestCase):
    def setUp(self):
        self._packed = gated_delta._ENABLE_GDN_PACKED
        gated_delta._ENABLE_GDN_PACKED = True

    def tearDown(self):
        gated_delta._ENABLE_GDN_PACKED = self._packed

    def test_packed_matches_explicit_tree(self):
        """Bitwise contract vs the comparator -- holds on any device."""
        for B, Hk, Hv, Dk, Dv, dtype in PACKED_CASES:
            for T in (1, 7, 64, 257, 2048):  # decode, ragged, aligned, long
                with self.subTest(B=B, Hk=Hk, Hv=Hv, Dv=Dv, dtype=dtype, T=T):
                    args = _make_inputs(B, T, Hk, Hv, Dk, Dv, dtype)
                    y_p, s_p = gated_delta_kernel_packed(*args)
                    y_x, s_x = gated_delta_kernel_xtree(*args)
                    mx.eval(y_p, s_p, y_x, s_x)
                    self.assertTrue(mx.array_equal(y_p, y_x))
                    self.assertTrue(mx.array_equal(s_p, s_x))

    def test_packed_matches_simd_sum_kernel(self):
        """Bitwise vs the kernel we actually replace. Measured to hold on M4.

        If this ever fails on a new device or toolchain the packed default
        should be revisited; the contract against the comparator still holds.
        """
        for B, Hk, Hv, Dk, Dv, dtype in PACKED_CASES:
            for T in (1, 64, 257, 2048):
                with self.subTest(B=B, Hk=Hk, Hv=Hv, Dv=Dv, dtype=dtype, T=T):
                    args = _make_inputs(B, T, Hk, Hv, Dk, Dv, dtype)
                    y_p, s_p = gated_delta_kernel_packed(*args)
                    y_u, s_u = gated_delta_kernel_unpacked(*args, None)
                    mx.eval(y_p, s_p, y_u, s_u)
                    self.assertTrue(mx.array_equal(y_p, y_u))
                    self.assertTrue(mx.array_equal(s_p, s_u))

    def test_explicit_tree_matches_simd_sum_kernel(self):
        """Canary: simd_sum lowers to the butterfly the comparator writes out.

        If this fails on a new device or toolchain the packed default should
        be revisited (its contract vs the comparator still holds).
        """
        args = _make_inputs(1, 257, 16, 32, 128, 128, mx.bfloat16)
        y_x, s_x = gated_delta_kernel_xtree(*args)
        y_u, s_u = gated_delta_kernel_unpacked(*args, None)
        mx.eval(y_x, s_x, y_u, s_u)
        self.assertTrue(mx.array_equal(y_x, y_u))
        self.assertTrue(mx.array_equal(s_x, s_u))

    def test_packed_matches_ops_reference(self):
        q, k, v, a, b, A_log, dt_bias, state = _make_inputs(
            1, 64, 16, 32, 128, 128, mx.bfloat16
        )
        y_p, s_p = gated_delta_kernel_packed(q, k, v, a, b, A_log, dt_bias, state)
        g = compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b.astype(mx.float32))
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
        mx.eval(y_p, s_p, y_r, s_r)
        self.assertLess(_rel_l2(y_p, y_r), 2e-3)
        self.assertLess(_rel_l2(s_p, s_r), 2e-3)

    def test_default_routing_uses_packed_when_eligible(self):
        args = _make_inputs(1, 64, 16, 32, 128, 128, mx.bfloat16)
        y_d, s_d = gated_delta_kernel(*args, None)
        y_p, s_p = gated_delta_kernel_packed(*args)
        mx.eval(y_d, s_d, y_p, s_p)
        self.assertTrue(mx.array_equal(y_d, y_p))
        self.assertTrue(mx.array_equal(s_d, s_p))

    def test_kill_switch_restores_original_kernel(self):
        # MLX_GDN_PACKED=0 routes back to the pre-existing simd_sum kernel.
        gated_delta._ENABLE_GDN_PACKED = False
        args = _make_inputs(1, 64, 16, 32, 128, 128, mx.bfloat16)
        y1, s1 = gated_delta_kernel(*args, None)
        y2, s2 = gated_delta_kernel_unpacked(*args, None)
        mx.eval(y1, s1, y2, s2)
        self.assertTrue(mx.array_equal(y1, y2))
        self.assertTrue(mx.array_equal(s1, s2))

    def test_unsupported_shapes_fall_back(self):
        # (Dk, Dv, use_mask)
        for Dk, Dv, use_mask in ((64, 64, False), (128, 128, True)):
            with self.subTest(Dk=Dk, Dv=Dv, use_mask=use_mask):
                args = _make_inputs(1, 32, 4, 8, Dk, Dv, mx.bfloat16)
                mask = mx.ones((1, 32), dtype=mx.bool_) if use_mask else None
                y_d, s_d = gated_delta_kernel(*args, mask)
                y_u, s_u = gated_delta_kernel_unpacked(*args, mask)
                mx.eval(y_d, s_d, y_u, s_u)
                self.assertTrue(mx.array_equal(y_d, y_u))
                self.assertTrue(mx.array_equal(s_d, s_u))

    def test_masked_generic_matches_ops_reference(self):
        """A ragged padding mask must match the ops reference at valid positions.

        test_unsupported_shapes_fall_back only proves the masked path routes to
        the generic kernel; its mask is all-ones, so it cannot detect a
        mask-handling bug. This one masks real positions off.
        """
        q, k, v, a, b, A_log, dt_bias, state = _make_inputs(
            2, 33, 8, 16, 128, 128, mx.bfloat16
        )
        mask = mx.arange(33)[None] < mx.array([[29], [17]])
        y_k, s_k = gated_delta_kernel(q, k, v, a, b, A_log, dt_bias, state, mask)
        g = compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b.astype(mx.float32))
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, mask)
        mx.eval(y_k, s_k, y_r, s_r)
        # Outputs at padded positions are unspecified (the kernel zeros them,
        # the ops reference does not); compare valid positions only.
        valid = mask[..., None, None]
        y_k = mx.where(valid, y_k, 0)
        y_r = mx.where(valid, y_r, 0)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_vector_gate_generic_matches_ops_reference(self):
        """A [B, T, Hv, Dk] gate takes the vectorized kernel; pin it to ops."""
        q, k, v, _, b, A_log, dt_bias, state = _make_inputs(
            1, 65, 4, 8, 128, 128, mx.bfloat16
        )
        a_vec = (mx.random.normal((1, 65, 8, 128)) * 0.5).astype(mx.bfloat16)
        mx.eval(a_vec)
        y_k, s_k = gated_delta_kernel(q, k, v, a_vec, b, A_log, dt_bias, state, None)
        # The vectorized kernel varies a per Dk element but still indexes A_log
        # and dt_bias per head, so broadcast them over Dk to match it.
        g = compute_g(A_log[:, None], a_vec, dt_bias[:, None])
        beta = mx.sigmoid(b.astype(mx.float32))
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
        mx.eval(y_k, s_k, y_r, s_r)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_small_head_dim_generic_matches_ops_reference(self):
        """Dk != 128 takes the generic kernel; pin it to ops, not just to itself."""
        q, k, v, a, b, A_log, dt_bias, state = _make_inputs(
            1, 65, 4, 8, 64, 64, mx.bfloat16
        )
        y_k, s_k = gated_delta_kernel(q, k, v, a, b, A_log, dt_bias, state, None)
        g = compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b.astype(mx.float32))
        y_r, s_r = gated_delta_ops(q, k, v, g, beta, state, None)
        mx.eval(y_k, s_k, y_r, s_r)
        self.assertLess(_rel_l2(y_k, y_r), 2e-3)
        self.assertLess(_rel_l2(s_k, s_r), 2e-3)

    def test_vector_gate_falls_back(self):
        """A [B, T, Hv, Dk] gate must keep the vectorized kernel."""
        q, k, v, _, b, A_log, dt_bias, state = _make_inputs(
            1, 65, 4, 8, 128, 128, mx.bfloat16
        )
        a_vec = (mx.random.normal((1, 65, 8, 128)) * 0.5).astype(mx.bfloat16)
        mx.eval(a_vec)
        args = (q, k, v, a_vec, b, A_log, dt_bias, state)
        y_d, s_d = gated_delta_kernel(*args, None)
        y_u, s_u = gated_delta_kernel_unpacked(*args, None)
        mx.eval(y_d, s_d, y_u, s_u)
        self.assertTrue(mx.array_equal(y_d, y_u))
        self.assertTrue(mx.array_equal(s_d, s_u))


if __name__ == "__main__":
    unittest.main()
