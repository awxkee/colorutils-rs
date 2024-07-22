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

use crate::avx::cie::{
    avx2_triple_to_lab, avx2_triple_to_luv, avx2_triple_to_xyz, avx_triple_to_lch,
};
use crate::avx::gamma_curves::get_avx2_linear_transfer;
use crate::avx::*;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::xyz_target::XyzTarget;

#[inline(always)]
pub unsafe fn avx2_image_to_xyz_lab<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    a_linearized: *mut f32,
    a_offset: usize,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> usize {
    if USE_ALPHA {
        if a_linearized.is_null() {
            panic!("Null alpha channel with requirements of linearized alpha if not supported");
        }
    }
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer = get_avx2_linear_transfer(transfer_function);

    let cq1 = _mm256_set1_ps(*matrix.get_unchecked(0).get_unchecked(0));
    let cq2 = _mm256_set1_ps(*matrix.get_unchecked(0).get_unchecked(1));
    let cq3 = _mm256_set1_ps(*matrix.get_unchecked(0).get_unchecked(2));
    let cq4 = _mm256_set1_ps(*matrix.get_unchecked(1).get_unchecked(0));
    let cq5 = _mm256_set1_ps(*matrix.get_unchecked(1).get_unchecked(1));
    let cq6 = _mm256_set1_ps(*matrix.get_unchecked(1).get_unchecked(2));
    let cq7 = _mm256_set1_ps(*matrix.get_unchecked(2).get_unchecked(0));
    let cq8 = _mm256_set1_ps(*matrix.get_unchecked(2).get_unchecked(1));
    let cq9 = _mm256_set1_ps(*matrix.get_unchecked(2).get_unchecked(2));

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 32 < width as usize {
        let (r_chan, g_chan, b_chan, a_chan);
        let src_ptr = src.add(src_offset + cx * channels);
        let row1 = _mm256_loadu_si256(src_ptr as *const __m256i);
        let row2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
        let row3 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (c1, c2, c3) = avx2_deinterleave_rgb_epi8(row1, row2, row3);
                if image_configuration == ImageConfiguration::Rgb {
                    r_chan = c1;
                    g_chan = c2;
                    b_chan = c3;
                } else {
                    r_chan = c3;
                    g_chan = c2;
                    b_chan = c1;
                }
                a_chan = _mm256_set1_epi8(0);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let row4 = _mm256_loadu_si256(src_ptr.add(64 + 32) as *const __m256i);
                let (c1, c2, c3, c4) = avx2_deinterleave_rgba_epi8(row1, row2, row3, row4);
                if image_configuration == ImageConfiguration::Rgba {
                    r_chan = c1;
                    g_chan = c2;
                    b_chan = c3;
                    a_chan = c4;
                } else {
                    r_chan = c3;
                    g_chan = c2;
                    b_chan = c1;
                    a_chan = c4;
                }
            }
        }

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (mut x_low_low, mut y_low_low, mut z_low_low) = avx2_triple_to_xyz(
            r_low_low, g_low_low, b_low_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9, &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = avx2_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = avx2_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
            XyzTarget::LCH => {
                let (l, c, h) = avx_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
        }

        let write_dst_ptr = dst_ptr.add(cx * 3);

        let (v0, v1, v2) = avx2_interleave_rgb_ps(x_low_low, y_low_low, z_low_low);

        _mm256_storeu_ps(write_dst_ptr, v0);
        _mm256_storeu_ps(write_dst_ptr.add(8), v1);
        _mm256_storeu_ps(write_dst_ptr.add(16), v2);

        let r_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r_low));
        let g_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(g_low));
        let b_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(b_low));

        let (mut x_low_high, mut y_low_high, mut z_low_high) = avx2_triple_to_xyz(
            r_low_high, g_low_high, b_low_high, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = avx2_triple_to_lab(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = a;
                z_low_high = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = avx2_triple_to_luv(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = u;
                z_low_high = v;
            }
            XyzTarget::LCH => {
                let (l, c, h) = avx_triple_to_lch(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = c;
                z_low_high = h;
            }
        }

        let (v0, v1, v2) = avx2_interleave_rgb_ps(x_low_high, y_low_high, z_low_high);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3), v0);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 + 8), v1);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 + 16), v2);

        let r_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_chan));
        let g_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_chan));
        let b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_chan));

        let r_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_high));
        let g_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_high));
        let b_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_high));

        let (mut x_high_low, mut y_high_low, mut z_high_low) = avx2_triple_to_xyz(
            r_high_low, g_high_low, b_high_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = avx2_triple_to_lab(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = a;
                z_high_low = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = avx2_triple_to_luv(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = u;
                z_high_low = v;
            }
            XyzTarget::LCH => {
                let (l, c, h) = avx_triple_to_lch(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = c;
                z_high_low = h;
            }
        }

        let (v0, v1, v2) = avx2_interleave_rgb_ps(x_high_low, y_high_low, z_high_low);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 2), v0);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 2 + 8), v1);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 2 + 16), v2);

        let r_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r_high));
        let g_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(g_high));
        let b_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(b_high));

        let (mut x_high_high, mut y_high_high, mut z_high_high) = avx2_triple_to_xyz(
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
            XyzTarget::LAB => {
                let (l, a, b) = avx2_triple_to_lab(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = a;
                z_high_high = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = avx2_triple_to_luv(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = u;
                z_high_high = v;
            }
            XyzTarget::LCH => {
                let (l, u, v) = avx_triple_to_lch(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = u;
                z_high_high = v;
            }
        }

        let (v0, v1, v2) = avx2_interleave_rgb_ps(x_high_high, y_high_high, z_high_high);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 3), v0);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 3 + 8), v1);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 3 + 16), v2);

        if USE_ALPHA {
            let a_ptr = (a_linearized as *mut u8).add(a_offset) as *mut f32;

            let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

            let u8_scale = _mm256_set1_ps(1f32 / 255f32);

            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx), a_low_low);

            let a_low_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a_low))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx + 8), a_low_high);

            let a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a_chan));

            let a_high_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_high))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx + 8 * 2), a_high_low);

            let a_high_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a_high))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx + 8 * 3), a_high_high);
        }

        cx += 32;
    }

    cx
}
