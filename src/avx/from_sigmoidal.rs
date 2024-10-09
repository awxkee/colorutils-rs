/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx::sigmoidal::avx_sigmoidal_to_rgb;
use crate::avx::{
    _mm256_packus_four_epi32, avx2_deinterleave_rgb_ps, avx2_deinterleave_rgba_ps,
    avx2_interleave_rgb, avx2_interleave_rgba_epi8,
};
use crate::avx_store_and_interleave_u8;
use crate::image::ImageConfiguration;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn vld_sigmoidal<const CHANNELS_CONFIGURATION: u8>(
    src: *const f32,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let v_scale_color = _mm256_set1_ps(255f32);
    let pixel_0 = _mm256_loadu_ps(src);
    let pixel_1 = _mm256_loadu_ps(src.add(8));
    let pixel_2 = _mm256_loadu_ps(src.add(16));
    if image_configuration.has_alpha() {
        let pixel_3 = _mm256_loadu_ps(src.add(24));
        let (sr, sg, sb, sa) = avx2_deinterleave_rgba_ps(pixel_0, pixel_1, pixel_2, pixel_3);

        let (r, g, b) = avx_sigmoidal_to_rgb(sr, sg, sb);
        let a_f32 = _mm256_mul_ps(sa, v_scale_color);
        (r, g, b, _mm256_cvtps_epi32(_mm256_round_ps::<0>(a_f32)))
    } else {
        let (sr, sg, sb) = avx2_deinterleave_rgb_ps(pixel_0, pixel_1, pixel_2);

        let (r, g, b) = avx_sigmoidal_to_rgb(sr, sg, sb);
        (r, g, b, _mm256_setzero_si256())
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn avx_from_sigmoidal_row<const CHANNELS_CONFIGURATION: u8>(
    start_cx: usize,
    src: *const f32,
    dst: *mut u8,
    width: u32,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let channels = image_configuration.get_channels_count();

    let mut cx = start_cx;

    while cx + 32 < width as usize {
        let offset_src_ptr = src.add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            vld_sigmoidal::<CHANNELS_CONFIGURATION>(src_ptr_0);

        let src_ptr_1 = offset_src_ptr.add(8 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            vld_sigmoidal::<CHANNELS_CONFIGURATION>(src_ptr_1);

        let src_ptr_2 = offset_src_ptr.add(8 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            vld_sigmoidal::<CHANNELS_CONFIGURATION>(src_ptr_2);

        let src_ptr_3 = offset_src_ptr.add(8 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            vld_sigmoidal::<CHANNELS_CONFIGURATION>(src_ptr_3);

        let r_row = _mm256_packus_four_epi32(r_row0_, r_row1_, r_row2_, r_row3_);
        let g_row = _mm256_packus_four_epi32(g_row0_, g_row1_, g_row2_, g_row3_);
        let b_row = _mm256_packus_four_epi32(b_row0_, b_row1_, b_row2_, b_row3_);
        let a_row = _mm256_packus_four_epi32(a_row0_, a_row1_, a_row2_, a_row3_);

        let dst_ptr = dst.add(cx * channels);

        avx_store_and_interleave_u8!(dst_ptr, image_configuration, r_row, g_row, b_row, a_row);
        cx += 32;
    }

    cx
}
