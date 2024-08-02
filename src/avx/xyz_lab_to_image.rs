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
use crate::avx::gamma_curves::get_avx_gamma_transfer;
use crate::avx::{
    _mm256_color_matrix_ps, avx2_deinterleave_rgb_ps, avx2_interleave_rgb,
    avx2_interleave_rgba_epi8, avx2_pack_u16, avx2_pack_u32,
};
use crate::image::ImageConfiguration;
use crate::xyz_target::XyzTarget;
use crate::{
    avx_store_and_interleave_v3_half_u8, avx_store_and_interleave_v3_quarter_u8,
    avx_store_and_interleave_v3_u8, avx_store_and_interleave_v4_half_u8,
    avx_store_and_interleave_v4_quarter_u8, avx_store_and_interleave_v4_u8, TransferFunction,
};

#[inline(always)]
unsafe fn avx_xyz_lab_vld<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: *const f32,
    transfer: &unsafe fn(__m256) -> __m256,
    c1: __m256,
    c2: __m256,
    c3: __m256,
    c4: __m256,
    c5: __m256,
    c6: __m256,
    c7: __m256,
    c8: __m256,
    c9: __m256,
) -> (__m256i, __m256i, __m256i) {
    let target: XyzTarget = TARGET.into();
    let v_scale_color = _mm256_set1_ps(255f32);
    let lab_pixel_0 = _mm256_loadu_ps(src);
    let lab_pixel_1 = _mm256_loadu_ps(src.add(8));
    let lab_pixel_2 = _mm256_loadu_ps(src.add(16));
    let (mut r_f32, mut g_f32, mut b_f32) =
        avx2_deinterleave_rgb_ps(lab_pixel_0, lab_pixel_1, lab_pixel_2);

    match target {
        XyzTarget::LAB => {
            let (x, y, z) = avx_lab_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::LUV => {
            let (x, y, z) = avx_luv_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::LCH => {
            let (x, y, z) = avx_lch_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        _ => {}
    }

    let (linear_r, linear_g, linear_b) =
        _mm256_color_matrix_ps(r_f32, g_f32, b_f32, c1, c2, c3, c4, c5, c6, c7, c8, c9);

    r_f32 = linear_r;
    g_f32 = linear_g;
    b_f32 = linear_b;

    r_f32 = transfer(r_f32);
    g_f32 = transfer(g_f32);
    b_f32 = transfer(b_f32);
    r_f32 = _mm256_mul_ps(r_f32, v_scale_color);
    g_f32 = _mm256_mul_ps(g_f32, v_scale_color);
    b_f32 = _mm256_mul_ps(b_f32, v_scale_color);
    (
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(r_f32)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(g_f32)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(b_f32)),
    )
}

#[inline(always)]
pub unsafe fn avx_xyz_to_channels<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
    const TRANSFER_FUNCTION: u8,
>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    a_channel: *const f32,
    a_offset: usize,
    dst: *mut u8,
    dst_offset: usize,
    width: u32,
    matrix: &[[f32; 3]; 3],
    _: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Alpha may be set only on images with alpha");
        }
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

    let transfer_function: TransferFunction = TRANSFER_FUNCTION.into();
    let transfer = get_avx_gamma_transfer(transfer_function);

    const CHANNELS: usize = 3usize;

    let color_rescale = _mm256_set1_ps(255f32);

    while cx + 32 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * CHANNELS);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            avx_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let src_ptr_1 = offset_src_ptr.add(8 * CHANNELS);

        let (r_row1_, g_row1_, b_row1_) =
            avx_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_1, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let src_ptr_2 = offset_src_ptr.add(8 * 2 * CHANNELS);

        let (r_row2_, g_row2_, b_row2_) =
            avx_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_2, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let src_ptr_3 = offset_src_ptr.add(8 * 3 * CHANNELS);

        let (r_row3_, g_row3_, b_row3_) =
            avx_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_3, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let r_row01 = avx2_pack_u32(r_row0_, r_row1_);
        let g_row01 = avx2_pack_u32(g_row0_, g_row1_);
        let b_row01 = avx2_pack_u32(b_row0_, b_row1_);

        let r_row23 = avx2_pack_u32(r_row2_, r_row3_);
        let g_row23 = avx2_pack_u32(g_row2_, g_row3_);
        let b_row23 = avx2_pack_u32(b_row2_, b_row3_);

        let r_row = avx2_pack_u16(r_row01, r_row23);
        let g_row = avx2_pack_u16(g_row01, g_row23);
        let b_row = avx2_pack_u16(b_row01, b_row23);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        if USE_ALPHA {
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_low_0_f = _mm256_loadu_ps(offset_a_src_ptr);
            let a_row0_ = _mm256_cvtps_epi32(_mm256_round_ps::<0>(_mm256_mul_ps(
                a_low_0_f,
                color_rescale,
            )));

            let a_low_1_f = _mm256_loadu_ps(offset_a_src_ptr.add(8));
            let a_row1_ = _mm256_cvtps_epi32(_mm256_round_ps::<0>(_mm256_mul_ps(
                a_low_1_f,
                color_rescale,
            )));

            let a_low_2_f = _mm256_loadu_ps(offset_a_src_ptr.add(16));
            let a_row2_ = _mm256_cvtps_epi32(_mm256_round_ps::<0>(_mm256_mul_ps(
                a_low_2_f,
                color_rescale,
            )));

            let a_low_3_f = _mm256_loadu_ps(offset_a_src_ptr.add(24));
            let a_row3_ = _mm256_cvtps_epi32(_mm256_round_ps::<0>(_mm256_mul_ps(
                a_low_3_f,
                color_rescale,
            )));

            let a_row01 = avx2_pack_u32(a_row0_, a_row1_);
            let a_row23 = avx2_pack_u32(a_row2_, a_row3_);
            let a_row = avx2_pack_u16(a_row01, a_row23);
            avx_store_and_interleave_v4_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row,
                a_row
            );
        } else {
            avx_store_and_interleave_v3_u8!(dst_ptr, image_configuration, r_row, g_row, b_row);
        }

        cx += 32;
    }

    let zeros = _mm256_setzero_si256();

    while cx + 16 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * CHANNELS);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            avx_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let src_ptr_1 = offset_src_ptr.add(8 * CHANNELS);

        let (r_row1_, g_row1_, b_row1_) =
            avx_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_1, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let r_row01 = avx2_pack_u32(r_row0_, r_row1_);
        let g_row01 = avx2_pack_u32(g_row0_, g_row1_);
        let b_row01 = avx2_pack_u32(b_row0_, b_row1_);

        let r_row = avx2_pack_u16(r_row01, zeros);
        let g_row = avx2_pack_u16(g_row01, zeros);
        let b_row = avx2_pack_u16(b_row01, zeros);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        if USE_ALPHA {
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_low_0_f = _mm256_loadu_ps(offset_a_src_ptr);
            let a_row0_ = _mm256_cvtps_epi32(_mm256_round_ps::<0>(_mm256_mul_ps(
                a_low_0_f,
                color_rescale,
            )));

            let a_low_1_f = _mm256_loadu_ps(offset_a_src_ptr.add(8));
            let a_row1_ = _mm256_cvtps_epi32(_mm256_round_ps::<0>(_mm256_mul_ps(
                a_low_1_f,
                color_rescale,
            )));

            let a_row01 = avx2_pack_u32(a_row0_, a_row1_);
            let a_row = avx2_pack_u16(a_row01, zeros);
            avx_store_and_interleave_v4_half_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row,
                a_row
            );
        } else {
            avx_store_and_interleave_v3_half_u8!(dst_ptr, image_configuration, r_row, g_row, b_row);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * CHANNELS);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            avx_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let r_row01 = avx2_pack_u32(r_row0_, zeros);
        let g_row01 = avx2_pack_u32(g_row0_, zeros);
        let b_row01 = avx2_pack_u32(b_row0_, zeros);

        let r_row = avx2_pack_u16(r_row01, zeros);
        let g_row = avx2_pack_u16(g_row01, zeros);
        let b_row = avx2_pack_u16(b_row01, zeros);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        if USE_ALPHA {
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_low_0_f = _mm256_loadu_ps(offset_a_src_ptr);
            let a_row0_ = _mm256_cvtps_epi32(_mm256_round_ps::<0>(_mm256_mul_ps(
                a_low_0_f,
                color_rescale,
            )));

            let a_row01 = avx2_pack_u32(a_row0_, zeros);
            let a_row = avx2_pack_u16(a_row01, zeros);
            avx_store_and_interleave_v4_quarter_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row,
                a_row
            );
        } else {
            avx_store_and_interleave_v3_quarter_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row
            );
        }

        cx += 8;
    }

    cx
}
