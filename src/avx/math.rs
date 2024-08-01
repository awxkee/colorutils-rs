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

use erydanos::{_mm256_pow_fast_ps, _mm256_prefer_fma_ps};

#[inline(always)]
pub unsafe fn _mm256_cube_ps(x: __m256) -> __m256 {
    _mm256_mul_ps(_mm256_mul_ps(x, x), x)
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_pow_ps(x: __m256, n: __m256) -> __m256 {
    _mm256_pow_fast_ps(x, n)
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_pow_n_ps(x: __m256, n: f32) -> __m256 {
    _mm256_pow_fast_ps(x, _mm256_set1_ps(n))
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn _mm256_fmaf_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_prefer_fma_ps(c, b, a)
}

#[inline(always)]
pub unsafe fn _mm256_cmpge_epi32(a: __m256i, b: __m256i) -> __m256i {
    let gt = _mm256_cmpgt_epi32(a, b);
    let eq = _mm256_cmpeq_epi32(a, b);
    return _mm256_or_si256(gt, eq);
}

#[inline(always)]
pub unsafe fn _mm256_cmplt_epi32(a: __m256i, b: __m256i) -> __m256i {
    return _mm256_cmpgt_epi32(b, a);
}

#[inline(always)]
pub unsafe fn _mm256_color_matrix_ps(
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
    let new_r = _mm256_prefer_fma_ps(_mm256_prefer_fma_ps(_mm256_mul_ps(g, c2), b, c3), r, c1);
    let new_g = _mm256_prefer_fma_ps(_mm256_prefer_fma_ps(_mm256_mul_ps(g, c5), b, c6), r, c4);
    let new_b = _mm256_prefer_fma_ps(_mm256_prefer_fma_ps(_mm256_mul_ps(g, c8), b, c9), r, c7);
    (new_r, new_g, new_b)
}
