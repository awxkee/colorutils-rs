/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::sse::*;
use crate::{
    load_f32_and_deinterleave, store_and_interleave_v3_half_u8, store_and_interleave_v3_u8,
    store_and_interleave_v4_half_u8, store_and_interleave_v4_u8, TransferFunction,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_gamma_vld<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: *const f32,
    transfer: &unsafe fn(__m128) -> __m128,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let v_scale_alpha = _mm_set1_ps(255f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let (mut r_f32, mut g_f32, mut b_f32, mut a_f32) =
        load_f32_and_deinterleave!(src, image_configuration);

    r_f32 = transfer(r_f32);
    g_f32 = transfer(g_f32);
    b_f32 = transfer(b_f32);
    r_f32 = _mm_mul_ps(r_f32, v_scale_alpha);
    g_f32 = _mm_mul_ps(g_f32, v_scale_alpha);
    b_f32 = _mm_mul_ps(b_f32, v_scale_alpha);
    if USE_ALPHA {
        a_f32 = _mm_mul_ps(a_f32, v_scale_alpha);
    }
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    if USE_ALPHA {
        (
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(a_f32)),
        )
    } else {
        (
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b_f32)),
            _mm_set1_epi32(255),
        )
    }
}

#[inline(always)]
pub unsafe fn sse_linear_to_gamma<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TRANSFER_FUNCTION: u8,
>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    _: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer_function: TransferFunction = TRANSFER_FUNCTION.into();
    let transfer = get_sse_gamma_transfer(transfer_function);

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, &transfer);

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_1, &transfer);

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_2, &transfer);

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_3, &transfer);

        let r_row01 = _mm_packus_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packus_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packus_epi32(b_row0_, b_row1_);

        let r_row23 = _mm_packus_epi32(r_row2_, r_row3_);
        let g_row23 = _mm_packus_epi32(g_row2_, g_row3_);
        let b_row23 = _mm_packus_epi32(b_row2_, b_row3_);

        let r_row = _mm_packus_epi16(r_row01, r_row23);
        let g_row = _mm_packus_epi16(g_row01, g_row23);
        let b_row = _mm_packus_epi16(b_row01, b_row23);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
            let a_row01 = _mm_packus_epi32(a_row0_, a_row1_);
            let a_row23 = _mm_packus_epi32(a_row2_, a_row3_);
            let a_row = _mm_packus_epi16(a_row01, a_row23);
            store_and_interleave_v4_u8!(dst_ptr, image_configuration, r_row, g_row, b_row, a_row);
        } else {
            store_and_interleave_v3_u8!(dst_ptr, image_configuration, r_row, g_row, b_row);
        }

        cx += 16;
    }

    let zeros = _mm_setzero_si128();

    while cx + 8 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, &transfer);

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_1, &transfer);

        let r_row01 = _mm_packus_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packus_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packus_epi32(b_row0_, b_row1_);

        let r_row = _mm_packus_epi16(r_row01, zeros);
        let g_row = _mm_packus_epi16(g_row01, zeros);
        let b_row = _mm_packus_epi16(b_row01, zeros);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
            let a_row01 = _mm_packus_epi32(a_row0_, a_row1_);
            let a_row = _mm_packus_epi16(a_row01, zeros);
            store_and_interleave_v4_half_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row,
                a_row
            );
        } else {
            store_and_interleave_v3_half_u8!(dst_ptr, image_configuration, r_row, g_row, b_row);
        }

        cx += 8;
    }

    cx
}
