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

use crate::avx::routines::avx_vld_u8_and_deinterleave;
use crate::avx::sigmoidal::avx_rgb_to_sigmoidal;
use crate::avx::{avx2_interleave_rgb_ps, avx2_interleave_rgba_ps};
use crate::image::ImageConfiguration;
use crate::{avx_store_and_interleave_v3_f32, avx_store_and_interleave_v4_f32};

#[inline]
pub unsafe fn avx_image_to_sigmoidal_row<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
>(
    start_cx: usize,
    src: *const u8,
    width: u32,
    dst: *mut f32,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let mut cx = start_cx;
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Use alpha flag used on image without alpha");
        }
    }

    let channels = image_configuration.get_channels_count();

    let dst_ptr = (dst as *mut u8) as *mut f32;

    let zeros = _mm256_setzero_si256();

    while cx + 32 < width as usize {
        let src_ptr = src.add(cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));
        let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (x_low_low, y_low_low, z_low_low) =
            avx_rgb_to_sigmoidal(r_low_low, g_low_low, b_low_low);

        let u8_scale = _mm256_set1_ps(1f32 / 255f32);

        if USE_ALPHA {
            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * channels);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_low,
                y_low_low,
                z_low_low,
                a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * channels);
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
            avx_rgb_to_sigmoidal(r_low_high, g_low_high, b_low_high);

        if USE_ALPHA {
            let a_low_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_low, zeros)),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * channels + 8 * channels);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_low_high,
                y_low_high,
                z_low_high,
                a_low_high
            );
        } else {
            let ptr = dst_ptr.add(cx * channels + 8 * channels);
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
            avx_rgb_to_sigmoidal(r_high_low, g_high_low, b_high_low);

        let a_high = _mm256_unpackhi_epi8(a_chan, zeros);

        if USE_ALPHA {
            let a_high_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_high))),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * channels + 8 * channels * 2);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_high_low,
                y_high_low,
                z_high_low,
                a_high_low
            );
        } else {
            let ptr = dst_ptr.add(cx * channels + 8 * channels * 2);
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
            avx_rgb_to_sigmoidal(r_high_high, g_high_high, b_high_high);

        if USE_ALPHA {
            let a_high_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_high, zeros)),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * channels + 8 * channels * 3);
            avx_store_and_interleave_v4_f32!(
                ptr,
                image_configuration,
                x_high_high,
                y_high_high,
                z_high_high,
                a_high_high
            );
        } else {
            let ptr = dst_ptr.add(cx * channels + 8 * channels * 3);
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

    cx
}
