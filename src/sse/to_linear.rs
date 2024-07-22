/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::load_u8_and_deinterleave;
use crate::sse::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_triple_to_linear(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    transfer: &unsafe fn(__m128) -> __m128,
) -> (__m128, __m128, __m128) {
    let u8_scale = _mm_set1_ps(1f32 / 255f32);
    let r_f = _mm_mul_ps(_mm_cvtepi32_ps(r), u8_scale);
    let g_f = _mm_mul_ps(_mm_cvtepi32_ps(g), u8_scale);
    let b_f = _mm_mul_ps(_mm_cvtepi32_ps(b), u8_scale);
    let r_linear = transfer(r_f);
    let g_linear = transfer(g_f);
    let b_linear = transfer(b_f);
    (r_linear, g_linear, b_linear)
}

#[inline(always)]
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

    let transfer = get_sse_linear_transfer(transfer_function);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

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
            sse_triple_to_linear(r_low_low, g_low_low, b_low_low, &transfer);

        let a_low = _mm_cvtepu8_epi16(a_chan);

        let u8_scale = _mm_set1_ps(1f32 / 255f32);

        if USE_ALPHA {
            let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);

            let (v0, v1, v2, v3) =
                sse_interleave_ps_rgba(x_low_low, y_low_low, z_low_low, a_low_low);
            _mm_storeu_ps(dst_ptr.add(cx * 4), v0);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 8), v2);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 12), v3);
        } else {
            let (v0, v1, v2) = sse_interleave_ps_rgb(x_low_low, y_low_low, z_low_low);
            _mm_storeu_ps(dst_ptr.add(cx * 3), v0);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 8), v2);
        }

        let r_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(r_low));
        let g_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(g_low));
        let b_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b_low));

        let (x_low_high, y_low_high, z_low_high) =
            sse_triple_to_linear(r_low_high, g_low_high, b_low_high, &transfer);

        if USE_ALPHA {
            let a_low_high = _mm_mul_ps(
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128::<8>(a_low))),
                u8_scale,
            );

            let (v0, v1, v2, v3) =
                sse_interleave_ps_rgba(x_low_high, y_low_high, z_low_high, a_low_high);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 16), v0);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 16 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 16 + 8), v2);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 16 + 12), v3);
        } else {
            let (v0, v1, v2) = sse_interleave_ps_rgb(x_low_high, y_low_high, z_low_high);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3), v0);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 + 8), v2);
        }

        let r_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(r_chan));
        let g_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(g_chan));
        let b_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(b_chan));

        let r_high_low = _mm_cvtepu16_epi32(r_high);
        let g_high_low = _mm_cvtepu16_epi32(g_high);
        let b_high_low = _mm_cvtepu16_epi32(b_high);

        let (x_high_low, y_high_low, z_high_low) =
            sse_triple_to_linear(r_high_low, g_high_low, b_high_low, &transfer);

        let a_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(a_chan));

        if USE_ALPHA {
            let a_high_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_high)), u8_scale);

            let (v0, v1, v2, v3) =
                sse_interleave_ps_rgba(x_high_low, y_high_low, z_high_low, a_high_low);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4 * 4 * 2), v0);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4 * 4 * 2 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4 * 4 * 2 + 8), v2);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4 * 4 * 2 + 12), v3);
        } else {
            let (v0, v1, v2) = sse_interleave_ps_rgb(x_high_low, y_high_low, z_high_low);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 2), v0);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 2 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 2 + 8), v2);
        }

        let r_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(r_high));
        let g_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(g_high));
        let b_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b_high));

        let (x_high_high, y_high_high, z_high_high) =
            sse_triple_to_linear(r_high_high, g_high_high, b_high_high, &transfer);

        if USE_ALPHA {
            let a_high_high = _mm_mul_ps(
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128::<8>(a_high))),
                u8_scale,
            );

            let (v0, v1, v2, v3) =
                sse_interleave_ps_rgba(x_high_high, y_high_high, z_high_high, a_high_high);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4 * 4 * 3), v0);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4 * 4 * 3 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4 * 4 * 3 + 8), v2);
            _mm_storeu_ps(dst_ptr.add(cx * 4 + 4 * 4 * 3 + 12), v3);
        } else {
            let (v0, v1, v2) = sse_interleave_ps_rgb(x_high_high, y_high_high, z_high_high);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 3), v0);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 3 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 3 + 8), v2);
        }

        cx += 16;
    }

    cx
}
