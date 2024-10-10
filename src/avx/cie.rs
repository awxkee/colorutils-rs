/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx::_mm256_cube_ps;
use crate::avx::math::*;
use crate::luv::{
    LUV_CUTOFF_FORWARD_Y, LUV_MULTIPLIER_FORWARD_Y, LUV_MULTIPLIER_INVERSE_Y, LUV_WHITE_U_PRIME,
    LUV_WHITE_V_PRIME,
};
use erydanos::{
    _mm256_atan2_ps, _mm256_cbrt_ps, _mm256_cos_ps, _mm256_hypot_ps, _mm256_prefer_fma_ps,
    _mm256_select_ps, _mm256_sin_ps,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn avx_lab_to_xyz(l: __m256, a: __m256, b: __m256) -> (__m256, __m256, __m256) {
    let y = _mm256_mul_ps(
        _mm256_add_ps(l, _mm256_set1_ps(16f32)),
        _mm256_set1_ps(1f32 / 116f32),
    );
    let x = _mm256_add_ps(_mm256_mul_ps(a, _mm256_set1_ps(1f32 / 500f32)), y);
    let z = _mm256_sub_ps(y, _mm256_mul_ps(b, _mm256_set1_ps(1f32 / 200f32)));
    let x3 = _mm256_cube_ps(x);
    let y3 = _mm256_cube_ps(y);
    let z3 = _mm256_cube_ps(z);
    let kappa = _mm256_set1_ps(0.008856f32);
    let k_sub = _mm256_set1_ps(16f32 / 116f32);
    let mult_1 = _mm256_set1_ps(1f32 / 7.787f32);
    let low_x = _mm256_mul_ps(_mm256_sub_ps(x, k_sub), mult_1);
    let low_y = _mm256_mul_ps(_mm256_sub_ps(y, k_sub), mult_1);
    let low_z = _mm256_mul_ps(_mm256_sub_ps(z, k_sub), mult_1);

    let x = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(x3, kappa), x3, low_x);
    let y = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(y3, kappa), y3, low_y);
    let z = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(z3, kappa), z3, low_z);
    let x = _mm256_mul_ps(x, _mm256_set1_ps(95.047f32 / 100f32));
    let z = _mm256_mul_ps(z, _mm256_set1_ps(108.883f32 / 100f32));
    (x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn avx_luv_to_xyz(l: __m256, u: __m256, v: __m256) -> (__m256, __m256, __m256) {
    let zeros = _mm256_setzero_ps();
    let zero_mask = _mm256_cmp_ps::<_CMP_EQ_OS>(l, zeros);
    let l13 = _mm256_rcp_ps(_mm256_mul_ps(l, _mm256_set1_ps(13f32)));
    let u = _mm256_prefer_fma_ps(_mm256_set1_ps(LUV_WHITE_U_PRIME), l13, u);
    let v = _mm256_prefer_fma_ps(_mm256_set1_ps(LUV_WHITE_V_PRIME), l13, v);
    let l_h = _mm256_mul_ps(
        _mm256_add_ps(l, _mm256_set1_ps(16f32)),
        _mm256_set1_ps(1f32 / 116f32),
    );
    let y_high = _mm256_mul_ps(_mm256_mul_ps(l_h, l_h), l_h);
    let y_low = _mm256_mul_ps(l, _mm256_set1_ps(LUV_MULTIPLIER_INVERSE_Y));
    let y = _mm256_select_ps(
        zero_mask,
        zeros,
        _mm256_select_ps(
            _mm256_cmp_ps::<_CMP_GT_OS>(l, _mm256_set1_ps(8f32)),
            y_high,
            y_low,
        ),
    );
    let zero_mask_2 = _mm256_cmp_ps::<_CMP_EQ_OS>(v, zeros);
    let den = _mm256_rcp_ps(_mm256_mul_ps(v, _mm256_set1_ps(4f32)));
    let mut x = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_mul_ps(y, u), den),
        _mm256_set1_ps(9f32),
    );
    x = _mm256_select_ps(zero_mask, zeros, x);
    x = _mm256_select_ps(zero_mask_2, zeros, x);
    let mut z = _mm256_mul_ps(
        _mm256_mul_ps(
            _mm256_prefer_fma_ps(
                _mm256_prefer_fma_ps(_mm256_set1_ps(12f32), _mm256_set1_ps(-3f32), u),
                v,
                _mm256_set1_ps(-20f32),
            ),
            y,
        ),
        den,
    );
    z = _mm256_select_ps(zero_mask, zeros, z);
    z = _mm256_select_ps(zero_mask_2, zeros, z);
    (x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn avx_lch_to_xyz(l: __m256, c: __m256, h: __m256) -> (__m256, __m256, __m256) {
    let u = _mm256_mul_ps(c, _mm256_cos_ps(h));
    let v = _mm256_mul_ps(c, _mm256_sin_ps(h));
    avx_luv_to_xyz(l, u, v)
}

#[inline(always)]
pub(crate) unsafe fn avx2_triple_to_xyz(
    r: __m256,
    g: __m256,
    b: __m256,
    c1: __m256,
    c2: __m256,
    c3: __m256,
    c4: __m256,
    c5: __m256,
    c6: __m256,
    c7: __m256,
    c8: __m256,
    c9: __m256,
) -> (__m256, __m256, __m256) {
    let (x, y, z) = _mm256_color_matrix_ps(r, g, b, c1, c2, c3, c4, c5, c6, c7, c8, c9);
    (x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn avx2_triple_to_luv(
    x: __m256,
    y: __m256,
    z: __m256,
) -> (__m256, __m256, __m256) {
    let zeros = _mm256_setzero_ps();
    let den = _mm256_prefer_fma_ps(
        _mm256_prefer_fma_ps(x, z, _mm256_set1_ps(3f32)),
        y,
        _mm256_set1_ps(15f32),
    );
    let nan_mask = _mm256_cmp_ps::<_CMP_LT_OS>(den, _mm256_set1_ps(0f32));
    let l_low_mask = _mm256_cmp_ps::<_CMP_LT_OS>(y, _mm256_set1_ps(LUV_CUTOFF_FORWARD_Y));
    let y_cbrt = _mm256_cbrt_ps(y);
    let l = _mm256_select_ps(
        l_low_mask,
        _mm256_mul_ps(y, _mm256_set1_ps(LUV_MULTIPLIER_FORWARD_Y)),
        _mm256_prefer_fma_ps(_mm256_set1_ps(-16f32), y_cbrt, _mm256_set1_ps(116f32)),
    );
    let u_prime = _mm256_div_ps(_mm256_mul_ps(x, _mm256_set1_ps(4f32)), den);
    let v_prime = _mm256_div_ps(_mm256_mul_ps(y, _mm256_set1_ps(9f32)), den);
    let sub_u_prime = _mm256_sub_ps(u_prime, _mm256_set1_ps(LUV_WHITE_U_PRIME));
    let sub_v_prime = _mm256_sub_ps(v_prime, _mm256_set1_ps(LUV_WHITE_V_PRIME));
    let l13 = _mm256_mul_ps(l, _mm256_set1_ps(13f32));
    let u = _mm256_select_ps(nan_mask, zeros, _mm256_mul_ps(l13, sub_u_prime));
    let v = _mm256_select_ps(nan_mask, zeros, _mm256_mul_ps(l13, sub_v_prime));
    (l, u, v)
}

#[inline(always)]
pub(crate) unsafe fn avx2_triple_to_lab(
    x: __m256,
    y: __m256,
    z: __m256,
) -> (__m256, __m256, __m256) {
    let x = _mm256_mul_ps(x, _mm256_set1_ps(100f32 / 95.047f32));
    let z = _mm256_mul_ps(z, _mm256_set1_ps(100f32 / 108.883f32));
    let cbrt_x = _mm256_cbrt_ps(x);
    let cbrt_y = _mm256_cbrt_ps(y);
    let cbrt_z = _mm256_cbrt_ps(z);
    let s_1 = _mm256_set1_ps(16.0 / 116.0);
    let s_2 = _mm256_set1_ps(7.787);
    let lower_x = _mm256_prefer_fma_ps(s_1, s_2, x);
    let lower_y = _mm256_prefer_fma_ps(s_1, s_2, y);
    let lower_z = _mm256_prefer_fma_ps(s_1, s_2, z);
    let cutoff = _mm256_set1_ps(0.008856f32);
    let x = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(x, cutoff), cbrt_x, lower_x);
    let y = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(y, cutoff), cbrt_y, lower_y);
    let z = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(z, cutoff), cbrt_z, lower_z);
    let l = _mm256_prefer_fma_ps(_mm256_set1_ps(-16.0f32), y, _mm256_set1_ps(116.0f32));
    let a = _mm256_mul_ps(_mm256_sub_ps(x, y), _mm256_set1_ps(500f32));
    let b = _mm256_mul_ps(_mm256_sub_ps(y, z), _mm256_set1_ps(200f32));
    (l, a, b)
}

#[inline(always)]
pub(crate) unsafe fn avx_triple_to_lch(
    x: __m256,
    y: __m256,
    z: __m256,
) -> (__m256, __m256, __m256) {
    let (luv_l, luv_u, luv_v) = avx2_triple_to_luv(x, y, z);
    let lch_c = _mm256_hypot_ps(luv_u, luv_v);
    let lch_h = _mm256_atan2_ps(luv_v, luv_u);
    (luv_l, lch_c, lch_h)
}
