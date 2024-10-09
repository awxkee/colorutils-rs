/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx::shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn avx2_interleave_rgb_ps(a: __m256, b: __m256, c: __m256) -> (__m256, __m256, __m256) {
    let b0 = _mm256_shuffle_epi32::<0x6c>(_mm256_castps_si256(a));
    let g0 = _mm256_shuffle_epi32::<0xb1>(_mm256_castps_si256(b));
    let r0 = _mm256_shuffle_epi32::<0xc6>(_mm256_castps_si256(c));

    let p0 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(b0, g0), r0);
    let p1 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(g0, r0), b0);
    let p2 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(r0, b0), g0);

    let bgr0 = _mm256_permute2x128_si256::<32>(p0, p1);
    let bgr2 = _mm256_permute2x128_si256::<49>(p0, p1);
    (
        _mm256_castsi256_ps(bgr0),
        _mm256_castsi256_ps(p2),
        _mm256_castsi256_ps(bgr2),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_packus_four_epi32(a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> __m256i {
    let ab = _mm256_packs_epi32(a, b);
    let cd = _mm256_packs_epi32(c, d);

    const MASK: i32 = shuffle(3, 1, 2, 0);

    let abcd = _mm256_permute4x64_epi64::<MASK>(_mm256_packus_epi16(ab, cd));
    _mm256_shuffle_epi32::<MASK>(abcd)
}
