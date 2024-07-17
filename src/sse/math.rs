/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use erydanos::_mm_pow_ps;

#[inline(always)]
pub unsafe fn _mm_cube_ps(x: __m128) -> __m128 {
    _mm_mul_ps(_mm_mul_ps(x, x), x)
}

#[cfg(not(target_feature = "fma"))]
#[inline]
pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    return _mm_add_ps(_mm_mul_ps(b, c), a);
}

#[cfg(target_feature = "fma")]
#[inline]
pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    return _mm_fmadd_ps(b, c, a);
}

#[inline(always)]
pub unsafe fn _mm_select_ps(mask: __m128, true_vals: __m128, false_vals: __m128) -> __m128 {
    _mm_blendv_ps(false_vals, true_vals, mask)
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_selecti_ps(mask: __m128i, true_vals: __m128, false_vals: __m128) -> __m128 {
    _mm_blendv_ps(false_vals, true_vals, _mm_castsi128_ps(mask))
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_select_si128(mask: __m128i, true_vals: __m128i, false_vals: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_and_si128(mask, true_vals),
        _mm_andnot_si128(mask, false_vals),
    )
}

#[inline(always)]
pub unsafe fn _mm_pow_n_ps(x: __m128, n: f32) -> __m128 {
    _mm_pow_ps(x, _mm_set1_ps(n))
}

#[inline(always)]
pub unsafe fn _mm_signbit_ps(f: __m128) -> __m128i {
    return _mm_and_si128(_mm_castps_si128(f), _mm_castps_si128(_mm_set1_ps(-0.0f32)));
}

#[inline(always)]
pub unsafe fn _mm_mulsign_ps(x: __m128, y: __m128) -> __m128 {
    return _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(x), _mm_signbit_ps(y)));
}

#[inline(always)]
pub unsafe fn _mm_pow2i_ps(q: __m128i) -> __m128 {
    return _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(q, _mm_set1_epi32(0x7f))));
}

#[inline(always)]
pub unsafe fn _mm_vldexp2_ps(d: __m128, e: __m128i) -> __m128 {
    return _mm_mul_ps(
        _mm_mul_ps(d, _mm_pow2i_ps(_mm_srli_epi32::<1>(e))),
        _mm_pow2i_ps(_mm_sub_epi32(e, _mm_srli_epi32::<1>(e))),
    );
}

#[inline(always)]
pub unsafe fn _mm_vilogbk_ps(d: __m128) -> __m128i {
    let o = _mm_cmplt_ps(d, _mm_set1_ps(5.421010862427522E-20f32));
    let d = _mm_select_ps(o, _mm_mul_ps(_mm_set1_ps(1.8446744073709552E19f32), d), d);
    let q = _mm_and_si128(
        _mm_srli_epi32::<23>(_mm_castps_si128(d)),
        _mm_set1_epi32(0xff),
    );
    let q = _mm_sub_epi32(
        q,
        _mm_select_si128(
            _mm_castps_si128(o),
            _mm_set1_epi32(64 + 0x7f),
            _mm_set1_epi32(0x7f),
        ),
    );
    return q;
}

#[inline(always)]
pub(crate) unsafe fn _mm_fmaf_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_prefer_fma_ps(c, b, a)
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn _mm_neg_epi32(x: __m128i) -> __m128i {
    let high = _mm_set1_epi32(0i32);
    return _mm_sub_epi32(high, x);
}

#[inline(always)]
pub(crate) unsafe fn _mm_neg_ps(x: __m128) -> __m128 {
    let high = _mm_set1_ps(0f32);
    return _mm_sub_ps(high, x);
}

#[inline(always)]
pub unsafe fn _mm_cmpge_epi32(a: __m128i, b: __m128i) -> __m128i {
    let gt = _mm_cmpgt_epi32(a, b);
    let eq = _mm_cmpeq_epi32(a, b);
    return _mm_or_si128(gt, eq);
}

#[inline(always)]
pub unsafe fn _mm_cmplt_epi32(a: __m128i, b: __m128i) -> __m128i {
    return _mm_cmpgt_epi32(b, a);
}

#[inline(always)]
pub unsafe fn _mm_color_matrix_ps(
    r: __m128,
    g: __m128,
    b: __m128,
    c1: __m128,
    c2: __m128,
    c3: __m128,
    c4: __m128,
    c5: __m128,
    c6: __m128,
    c7: __m128,
    c8: __m128,
    c9: __m128,
) -> (__m128, __m128, __m128) {
    let new_r = _mm_prefer_fma_ps(_mm_prefer_fma_ps(_mm_mul_ps(g, c2), b, c3), r, c1);
    let new_g = _mm_prefer_fma_ps(_mm_prefer_fma_ps(_mm_mul_ps(g, c5), b, c6), r, c4);
    let new_b = _mm_prefer_fma_ps(_mm_prefer_fma_ps(_mm_mul_ps(g, c8), b, c9), r, c7);
    (new_r, new_g, new_b)
}
