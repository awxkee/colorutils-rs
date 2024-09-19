/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::sse::cie::{sse_triple_to_lab, sse_triple_to_lch, sse_triple_to_luv, sse_triple_to_xyz};
use crate::sse::*;
use crate::xyz_target::XyzTarget;
use crate::{load_u8_and_deinterleave, load_u8_and_deinterleave_half, store_and_interleave_v4_f32};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn sse_channels_to_xyza_laba<
    const CHANNELS_CONFIGURATION: u8,
    const TARGET: u8,
    const TRANSFER_FUNCTION: u8,
>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    matrix: &[[f32; 3]; 3],
    _: TransferFunction,
) -> usize {
    const CHANNELS: usize = 4;
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    if !image_configuration.has_alpha() {
        panic!("Null alpha channel with requirements of linearized alpha if not supported");
    }
    let mut cx = start_cx;

    let transfer_function: TransferFunction = TRANSFER_FUNCTION.into();
    let transfer = get_sse_linear_transfer(transfer_function);

    let cq1 = _mm_set1_ps(*matrix.get_unchecked(0).get_unchecked(0));
    let cq2 = _mm_set1_ps(*matrix.get_unchecked(0).get_unchecked(1));
    let cq3 = _mm_set1_ps(*matrix.get_unchecked(0).get_unchecked(2));
    let cq4 = _mm_set1_ps(*matrix.get_unchecked(1).get_unchecked(0));
    let cq5 = _mm_set1_ps(*matrix.get_unchecked(1).get_unchecked(1));
    let cq6 = _mm_set1_ps(*matrix.get_unchecked(1).get_unchecked(2));
    let cq7 = _mm_set1_ps(*matrix.get_unchecked(2).get_unchecked(0));
    let cq8 = _mm_set1_ps(*matrix.get_unchecked(2).get_unchecked(1));
    let cq9 = _mm_set1_ps(*matrix.get_unchecked(2).get_unchecked(2));

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

        let (mut x_low_low, mut y_low_low, mut z_low_low) = sse_triple_to_xyz(
            r_low_low, g_low_low, b_low_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9, &transfer,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = sse_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = sse_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = sse_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
        }

        let a_low = _mm_cvtepu8_epi16(a_chan);
        let u8_scale = _mm_set1_ps(1f32 / 255f32);
        let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);

        let (v0, v1, v2, v3) = sse_interleave_ps_rgba(x_low_low, y_low_low, z_low_low, a_low_low);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 12), v3);

        let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

        let (mut x_low_high, mut y_low_high, mut z_low_high) = sse_triple_to_xyz(
            r_low_high, g_low_high, b_low_high, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = sse_triple_to_lab(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = a;
                z_low_high = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = sse_triple_to_luv(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = u;
                z_low_high = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = sse_triple_to_lch(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = c;
                z_low_high = h;
            }
        }

        let a_low_high = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_low, zeros)), u8_scale);

        let ptr0 = dst_ptr.add(cx * CHANNELS + 4 * CHANNELS);
        store_and_interleave_v4_f32!(
            ptr0,
            image_configuration,
            x_low_high,
            y_low_high,
            z_low_high,
            a_low_high
        );

        let r_high = _mm_unpackhi_epi8(r_chan, _mm_setzero_si128());
        let g_high = _mm_unpackhi_epi8(g_chan, _mm_setzero_si128());
        let b_high = _mm_unpackhi_epi8(b_chan, _mm_setzero_si128());

        let r_high_low = _mm_cvtepu16_epi32(r_high);
        let g_high_low = _mm_cvtepu16_epi32(g_high);
        let b_high_low = _mm_cvtepu16_epi32(b_high);

        let (mut x_high_low, mut y_high_low, mut z_high_low) = sse_triple_to_xyz(
            r_high_low, g_high_low, b_high_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = sse_triple_to_lab(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = a;
                z_high_low = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = sse_triple_to_luv(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = u;
                z_high_low = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = sse_triple_to_lch(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = c;
                z_high_low = h;
            }
        }

        let a_high = _mm_unpackhi_epi8(a_chan, _mm_setzero_si128());

        let a_high_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_high)), u8_scale);

        let (v0, v1, v2, v3) =
            sse_interleave_ps_rgba(x_high_low, y_high_low, z_high_low, a_high_low);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 2), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 2 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 2 + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 2 + 12), v3);

        let r_high_high = _mm_unpackhi_epi16(r_high, _mm_setzero_si128());
        let g_high_high = _mm_unpackhi_epi16(g_high, _mm_setzero_si128());
        let b_high_high = _mm_unpackhi_epi16(b_high, _mm_setzero_si128());

        let (mut x_high_high, mut y_high_high, mut z_high_high) = sse_triple_to_xyz(
            r_high_high,
            g_high_high,
            b_high_high,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            &transfer,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = sse_triple_to_lab(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = a;
                z_high_high = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = sse_triple_to_luv(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = u;
                z_high_high = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = sse_triple_to_lch(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = c;
                z_high_high = h;
            }
        }

        let a_high_high = _mm_mul_ps(
            _mm_cvtepi32_ps(_mm_unpackhi_epi16(a_high, _mm_setzero_si128())),
            u8_scale,
        );

        let (v0, v1, v2, v3) =
            sse_interleave_ps_rgba(x_high_high, y_high_high, z_high_high, a_high_high);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 3), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 3 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 3 + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 3 + 12), v3);

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

        let (mut x_low_low, mut y_low_low, mut z_low_low) = sse_triple_to_xyz(
            r_low_low, g_low_low, b_low_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9, &transfer,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = sse_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = sse_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = sse_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
        }

        let a_low = _mm_cvtepu8_epi16(a_chan);
        let u8_scale = _mm_set1_ps(1f32 / 255f32);
        let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);

        let (v0, v1, v2, v3) = sse_interleave_ps_rgba(x_low_low, y_low_low, z_low_low, a_low_low);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 12), v3);

        let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

        let (mut x_low_high, mut y_low_high, mut z_low_high) = sse_triple_to_xyz(
            r_low_high, g_low_high, b_low_high, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = sse_triple_to_lab(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = a;
                z_low_high = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = sse_triple_to_luv(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = u;
                z_low_high = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = sse_triple_to_lch(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = c;
                z_low_high = h;
            }
        }

        let a_low_high = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_low, zeros)), u8_scale);

        let ptr0 = dst_ptr.add(cx * CHANNELS + 4 * CHANNELS);
        store_and_interleave_v4_f32!(
            ptr0,
            image_configuration,
            x_low_high,
            y_low_high,
            z_low_high,
            a_low_high
        );

        cx += 8;
    }

    cx
}
