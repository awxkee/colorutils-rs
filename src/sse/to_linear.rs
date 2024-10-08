/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::sse::*;
use crate::{
    load_u8_and_deinterleave, load_u8_and_deinterleave_half, store_and_interleave_v3_f32,
    store_and_interleave_v4_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_triple_to_linear(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    transfer_function: TransferFunction,
) -> (__m128, __m128, __m128) {
    let u8_scale = _mm_set1_ps(1f32 / 255f32);
    let r_f = _mm_mul_ps(_mm_cvtepi32_ps(r), u8_scale);
    let g_f = _mm_mul_ps(_mm_cvtepi32_ps(g), u8_scale);
    let b_f = _mm_mul_ps(_mm_cvtepi32_ps(b), u8_scale);
    let r_linear = perform_sse_linear_transfer(transfer_function, r_f);
    let g_linear = perform_sse_linear_transfer(transfer_function, g_f);
    let b_linear = perform_sse_linear_transfer(transfer_function, b_f);
    (r_linear, g_linear, b_linear)
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_channels_to_linear<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    let zeros = _mm_setzero_si128();

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_u8_and_deinterleave!(src_ptr, image_configuration);

        let r_low = _mm_cvtepu8_epi16(r_chan);
        let g_low = _mm_cvtepu8_epi16(g_chan);
        let b_low = _mm_cvtepu8_epi16(b_chan);

        let r_low_low = _mm_cvtepu16_epi32(r_low);
        let g_low_low = _mm_cvtepu16_epi32(g_low);
        let b_low_low = _mm_cvtepu16_epi32(b_low);

        let (x_low_low, y_low_low, z_low_low) =
            sse_triple_to_linear(r_low_low, g_low_low, b_low_low, transfer_function);

        let a_low = _mm_cvtepu8_epi16(a_chan);

        let u8_scale = _mm_set1_ps(1f32 / 255f32);

        if USE_ALPHA {
            let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);

            let ptr = dst_ptr.add(cx * 4);
            store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low,
                a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3);
            store_and_interleave_v3_f32!(ptr, image_configuration, x_low_low, y_low_low, z_low_low);
        }

        let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) =
            sse_triple_to_linear(r_low_high, g_low_high, b_low_high, transfer_function);

        if USE_ALPHA {
            let a_low_high =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_low, zeros)), u8_scale);

            let ptr = dst_ptr.add(cx * 4 + 16);
            store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high,
                a_low_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 4 * 3);
            store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high
            );
        }

        let r_high = _mm_unpackhi_epi8(r_chan, zeros);
        let g_high = _mm_unpackhi_epi8(g_chan, zeros);
        let b_high = _mm_unpackhi_epi8(b_chan, zeros);

        let r_high_low = _mm_cvtepu16_epi32(r_high);
        let g_high_low = _mm_cvtepu16_epi32(g_high);
        let b_high_low = _mm_cvtepu16_epi32(b_high);

        let (x_high_low, y_high_low, z_high_low) =
            sse_triple_to_linear(r_high_low, g_high_low, b_high_low, transfer_function);

        let a_high = _mm_unpackhi_epi8(a_chan, zeros);

        if USE_ALPHA {
            let a_high_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_high)), u8_scale);

            let ptr = dst_ptr.add(cx * 4 + 4 * 4 * 2);
            store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_high_low,
                y_high_low,
                z_high_low,
                a_high_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 4 * 3 * 2);
            store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_high_low,
                y_high_low,
                z_high_low
            );
        }

        let r_high_high = _mm_unpackhi_epi16(r_high, zeros);
        let g_high_high = _mm_unpackhi_epi16(g_high, zeros);
        let b_high_high = _mm_unpackhi_epi16(b_high, zeros);

        let (x_high_high, y_high_high, z_high_high) =
            sse_triple_to_linear(r_high_high, g_high_high, b_high_high, transfer_function);

        if USE_ALPHA {
            let a_high_high = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_high)), u8_scale);

            let ptr = dst_ptr.add(cx * 4 + 4 * 4 * 3);
            store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_high_high,
                y_high_high,
                z_high_high,
                a_high_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 4 * 3 * 3);
            store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_high_high,
                y_high_high,
                z_high_high
            );
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_u8_and_deinterleave_half!(src_ptr, image_configuration);

        let r_low = _mm_cvtepu8_epi16(r_chan);
        let g_low = _mm_cvtepu8_epi16(g_chan);
        let b_low = _mm_cvtepu8_epi16(b_chan);

        let r_low_low = _mm_cvtepu16_epi32(r_low);
        let g_low_low = _mm_cvtepu16_epi32(g_low);
        let b_low_low = _mm_cvtepu16_epi32(b_low);

        let (x_low_low, y_low_low, z_low_low) =
            sse_triple_to_linear(r_low_low, g_low_low, b_low_low, transfer_function);

        let a_low = _mm_cvtepu8_epi16(a_chan);

        let u8_scale = _mm_set1_ps(1f32 / 255f32);

        if USE_ALPHA {
            let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);

            let ptr = dst_ptr.add(cx * 4);
            store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low,
                a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3);
            store_and_interleave_v3_f32!(ptr, image_configuration, x_low_low, y_low_low, z_low_low);
        }

        let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) =
            sse_triple_to_linear(r_low_high, g_low_high, b_low_high, transfer_function);

        if USE_ALPHA {
            let a_low_high =
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_low, zeros)), u8_scale);

            let ptr = dst_ptr.add(cx * 4 + 16);
            store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high,
                a_low_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 4 * 3);
            store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high
            );
        }

        cx += 8;
    }

    cx
}
