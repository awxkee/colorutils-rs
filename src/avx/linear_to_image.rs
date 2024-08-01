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

use crate::avx::gamma_curves::get_avx_gamma_transfer;
use crate::avx::routines::avx_vld_f32_and_deinterleave;
use crate::avx::{avx2_interleave_rgb, avx2_interleave_rgba_epi8, avx2_pack_s32, avx2_pack_u16};
use crate::image::ImageConfiguration;
use crate::{
    avx_store_and_interleave_v3_half_u8, avx_store_and_interleave_v3_u8,
    avx_store_and_interleave_v4_half_u8, avx_store_and_interleave_v4_u8, TransferFunction,
};

#[inline(always)]
unsafe fn gamma_vld<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: *const f32,
    transfer_function: TransferFunction,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let transfer = get_avx_gamma_transfer(transfer_function);
    let v_scale_alpha = _mm256_set1_ps(255f32);
    let (mut r_f32, mut g_f32, mut b_f32, mut a_f32) =
        avx_vld_f32_and_deinterleave::<CHANNELS_CONFIGURATION>(src);
    r_f32 = transfer(r_f32);
    g_f32 = transfer(g_f32);
    b_f32 = transfer(b_f32);
    r_f32 = _mm256_mul_ps(r_f32, v_scale_alpha);
    g_f32 = _mm256_mul_ps(g_f32, v_scale_alpha);
    b_f32 = _mm256_mul_ps(b_f32, v_scale_alpha);
    if USE_ALPHA {
        a_f32 = _mm256_mul_ps(a_f32, v_scale_alpha);
    }
    (
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(r_f32)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(g_f32)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(b_f32)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(a_f32)),
    )
}

#[inline(always)]
pub unsafe fn avx_linear_to_gamma<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    while cx + 32 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, transfer_function);

        let src_ptr_1 = offset_src_ptr.add(8 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_1, transfer_function);

        let src_ptr_2 = offset_src_ptr.add(8 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_2, transfer_function);

        let src_ptr_3 = offset_src_ptr.add(8 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_3, transfer_function);

        let r_row01 = avx2_pack_s32(r_row0_, r_row1_);
        let g_row01 = avx2_pack_s32(g_row0_, g_row1_);
        let b_row01 = avx2_pack_s32(b_row0_, b_row1_);

        let r_row23 = avx2_pack_s32(r_row2_, r_row3_);
        let g_row23 = avx2_pack_s32(g_row2_, g_row3_);
        let b_row23 = avx2_pack_s32(b_row2_, b_row3_);

        let r_row = avx2_pack_u16(r_row01, r_row23);
        let g_row = avx2_pack_u16(g_row01, g_row23);
        let b_row = avx2_pack_u16(b_row01, b_row23);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
            let a_row01 = avx2_pack_s32(a_row0_, a_row1_);
            let a_row23 = avx2_pack_s32(a_row2_, a_row3_);
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
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, transfer_function);

        let src_ptr_1 = offset_src_ptr.add(8 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_1, transfer_function);

        let r_row01 = avx2_pack_s32(r_row0_, r_row1_);
        let g_row01 = avx2_pack_s32(g_row0_, g_row1_);
        let b_row01 = avx2_pack_s32(b_row0_, b_row1_);

        let r_row = avx2_pack_u16(r_row01, zeros);
        let g_row = avx2_pack_u16(g_row01, zeros);
        let b_row = avx2_pack_u16(b_row01, zeros);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
            let a_row01 = avx2_pack_s32(a_row0_, a_row1_);
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

    cx
}
