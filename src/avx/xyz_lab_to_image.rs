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

use crate::avx::cie::{avx_lab_to_xyz, avx_lch_to_xyz, avx_luv_to_xyz};
use crate::avx::{_mm256_color_matrix_ps, avx2_deinterleave_rgb_ps};
use crate::avx::{avx2_interleave_rgb_ps, avx2_interleave_rgba_ps};
use crate::image::ImageConfiguration;
use crate::sse::sse_xyz_lab_vld;
use crate::sse::{sse_interleave_ps_rgb, sse_interleave_ps_rgba};
use crate::xyz_target::XyzTarget;
use crate::{
    avx_store_and_interleave_v3_f32, avx_store_and_interleave_v4_f32, store_and_interleave_v3_f32,
    store_and_interleave_v4_f32,
};

#[inline(always)]
unsafe fn avx_xyz_lab_vld<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: *const f32,
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
    let target: XyzTarget = TARGET.into();
    let lab_pixel_0 = _mm256_loadu_ps(src);
    let lab_pixel_1 = _mm256_loadu_ps(src.add(8));
    let lab_pixel_2 = _mm256_loadu_ps(src.add(16));
    let (mut r_f32, mut g_f32, mut b_f32) =
        avx2_deinterleave_rgb_ps(lab_pixel_0, lab_pixel_1, lab_pixel_2);

    match target {
        XyzTarget::Lab => {
            let (x, y, z) = avx_lab_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::Luv => {
            let (x, y, z) = avx_luv_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::Lch => {
            let (x, y, z) = avx_lch_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        _ => {}
    }

    let (linear_r, linear_g, linear_b) =
        _mm256_color_matrix_ps(r_f32, g_f32, b_f32, c1, c2, c3, c4, c5, c6, c7, c8, c9);

    (linear_r, linear_g, linear_b)
}

#[target_feature(enable = "avx2")]
pub unsafe fn avx_xyz_to_channels<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    a_channel: *const f32,
    a_offset: usize,
    dst: *mut f32,
    dst_offset: usize,
    width: u32,
    matrix: &[[f32; 3]; 3],
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut cx = start_cx;

    let c1 = _mm256_set1_ps(*matrix.get_unchecked(0).get_unchecked(0));
    let c2 = _mm256_set1_ps(*matrix.get_unchecked(0).get_unchecked(1));
    let c3 = _mm256_set1_ps(*matrix.get_unchecked(0).get_unchecked(2));
    let c4 = _mm256_set1_ps(*matrix.get_unchecked(1).get_unchecked(0));
    let c5 = _mm256_set1_ps(*matrix.get_unchecked(1).get_unchecked(1));
    let c6 = _mm256_set1_ps(*matrix.get_unchecked(1).get_unchecked(2));
    let c7 = _mm256_set1_ps(*matrix.get_unchecked(2).get_unchecked(0));
    let c8 = _mm256_set1_ps(*matrix.get_unchecked(2).get_unchecked(1));
    let c9 = _mm256_set1_ps(*matrix.get_unchecked(2).get_unchecked(2));

    const CHANNELS: usize = 3usize;

    while cx + 8 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * CHANNELS);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            avx_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let dst_ptr = ((dst as *mut u8).add(dst_offset) as *mut f32).add(cx * channels);

        if USE_ALPHA {
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_row = _mm256_loadu_ps(offset_a_src_ptr);

            avx_store_and_interleave_v4_f32!(
                dst_ptr,
                image_configuration,
                r_row0_,
                g_row0_,
                b_row0_,
                a_row
            );
        } else {
            avx_store_and_interleave_v3_f32!(
                dst_ptr,
                image_configuration,
                r_row0_,
                g_row0_,
                b_row0_
            );
        }

        cx += 8;
    }

    while cx + 4 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            sse_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0,
                _mm256_castps256_ps128(c1),
                _mm256_castps256_ps128(c2),
                _mm256_castps256_ps128(c3),
                _mm256_castps256_ps128(c4),
                _mm256_castps256_ps128(c5),
                _mm256_castps256_ps128(c6),
                _mm256_castps256_ps128(c7),
                _mm256_castps256_ps128(c8),
                _mm256_castps256_ps128(c9),
            );

        let dst_ptr = ((dst as *mut u8).add(dst_offset) as *mut f32).add(cx * channels);

        if USE_ALPHA {
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_row = _mm_loadu_ps(offset_a_src_ptr);

            store_and_interleave_v4_f32!(
                dst_ptr,
                image_configuration,
                r_row0_,
                g_row0_,
                b_row0_,
                a_row
            );
        } else {
            store_and_interleave_v3_f32!(dst_ptr, image_configuration, r_row0_, g_row0_, b_row0_);
        }

        cx += 4;
    }

    cx
}
