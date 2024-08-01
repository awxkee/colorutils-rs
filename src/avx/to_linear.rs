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

use crate::avx::gamma_curves::get_avx2_linear_transfer;
use crate::avx::routines::{
    avx_vld_u8_and_deinterleave, avx_vld_u8_and_deinterleave_half,
    avx_vld_u8_and_deinterleave_quarter,
};
use crate::avx::{avx2_interleave_rgb_ps, avx2_interleave_rgba_ps};
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::{avx_store_and_interleave_v3_f32, avx_store_and_interleave_v4_f32};

#[inline(always)]
unsafe fn triple_to_linear(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    transfer: &unsafe fn(__m256) -> __m256,
) -> (__m256, __m256, __m256) {
    let u8_scale = _mm256_set1_ps(1f32 / 255f32);
    let r_f = _mm256_mul_ps(_mm256_cvtepi32_ps(r), u8_scale);
    let g_f = _mm256_mul_ps(_mm256_cvtepi32_ps(g), u8_scale);
    let b_f = _mm256_mul_ps(_mm256_cvtepi32_ps(b), u8_scale);
    let r_linear = transfer(r_f);
    let g_linear = transfer(g_f);
    let b_linear = transfer(b_f);
    (r_linear, g_linear, b_linear)
}

#[inline(always)]
pub unsafe fn avx_channels_to_linear<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
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

    let transfer = get_avx2_linear_transfer(transfer_function);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    let zeros = _mm256_setzero_si256();

    while cx + 32 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (x_low_low, y_low_low, z_low_low) =
            triple_to_linear(r_low_low, g_low_low, b_low_low, &transfer);

        let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

        let u8_scale = _mm256_set1_ps(1f32 / 255f32);

        if USE_ALPHA {
            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low,
                a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3);
            avx_store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low
            );
        }

        let r_low_high = _mm256_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm256_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm256_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) =
            triple_to_linear(r_low_high, g_low_high, b_low_high, &transfer);

        if USE_ALPHA {
            let a_low_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_low, zeros)),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4 + 32);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high,
                a_low_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 24);
            avx_store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high
            );
        }

        let r_high = _mm256_unpackhi_epi8(r_chan, zeros);
        let g_high = _mm256_unpackhi_epi8(g_chan, zeros);
        let b_high = _mm256_unpackhi_epi8(b_chan, zeros);

        let r_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_high));
        let g_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_high));
        let b_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_high));

        let (x_high_low, y_high_low, z_high_low) =
            triple_to_linear(r_high_low, g_high_low, b_high_low, &transfer);

        let a_high = _mm256_unpackhi_epi8(a_chan, zeros);

        if USE_ALPHA {
            let a_high_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_high))),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4 + 64);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_high_low,
                y_high_low,
                z_high_low,
                a_high_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 48);
            avx_store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_high_low,
                y_high_low,
                z_high_low
            );
        }

        let r_high_high = _mm256_unpackhi_epi16(r_high, zeros);
        let g_high_high = _mm256_unpackhi_epi16(g_high, zeros);
        let b_high_high = _mm256_unpackhi_epi16(b_high, zeros);

        let (x_high_high, y_high_high, z_high_high) =
            triple_to_linear(r_high_high, g_high_high, b_high_high, &transfer);

        if USE_ALPHA {
            let a_high_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_high, zeros)),
                u8_scale,
            );
            let ptr = dst_ptr.add(cx * 4 + 96);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_high_high,
                y_high_high,
                z_high_high,
                a_high_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 24 * 3);
            avx_store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_high_high,
                y_high_high,
                z_high_high
            );
        }

        cx += 32;
    }

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave_half::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (x_low_low, y_low_low, z_low_low) =
            triple_to_linear(r_low_low, g_low_low, b_low_low, &transfer);

        let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

        let u8_scale = _mm256_set1_ps(1f32 / 255f32);

        if USE_ALPHA {
            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low,
                a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3);
            avx_store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low
            );
        }

        let r_low_high = _mm256_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm256_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm256_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) =
            triple_to_linear(r_low_high, g_low_high, b_low_high, &transfer);

        if USE_ALPHA {
            let a_low_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_low, zeros)),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4 + 32);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high,
                a_low_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 24);
            avx_store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high
            );
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave_quarter::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (x_low_low, y_low_low, z_low_low) =
            triple_to_linear(r_low_low, g_low_low, b_low_low, &transfer);

        let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

        let u8_scale = _mm256_set1_ps(1f32 / 255f32);

        if USE_ALPHA {
            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low,
                a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3);
            avx_store_and_interleave_v3_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low
            );
        }
        cx += 8;
    }

    cx
}
