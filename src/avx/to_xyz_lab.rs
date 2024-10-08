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
use crate::avx::routines::{avx_vld_u8_and_deinterleave, avx_vld_u8_and_deinterleave_half};
use crate::avx::*;
use crate::avx_store_and_interleave_v3_direct_f32;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::xyz_target::XyzTarget;

#[target_feature(enable = "avx2")]
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
    if USE_ALPHA && a_linearized.is_null() {
        panic!("Null alpha channel with requirements of linearized alpha if not supported");
    }
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

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
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (mut x_low_low, mut y_low_low, mut z_low_low) = avx2_triple_to_xyz(
            r_low_low,
            g_low_low,
            b_low_low,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            transfer_function,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = avx2_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = avx2_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = avx_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
        }

        let write_dst_ptr = dst_ptr.add(cx * 3);
        avx_store_and_interleave_v3_direct_f32!(write_dst_ptr, x_low_low, y_low_low, z_low_low);

        let r_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r_low));
        let g_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(g_low));
        let b_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(b_low));

        let (mut x_low_high, mut y_low_high, mut z_low_high) = avx2_triple_to_xyz(
            r_low_high,
            g_low_high,
            b_low_high,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            transfer_function,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = avx2_triple_to_lab(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = a;
                z_low_high = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = avx2_triple_to_luv(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = u;
                z_low_high = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = avx_triple_to_lch(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = c;
                z_low_high = h;
            }
        }

        let ptr2 = write_dst_ptr.add(8 * 3);
        avx_store_and_interleave_v3_direct_f32!(ptr2, x_low_high, y_low_high, z_low_high);

        let r_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_chan));
        let g_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_chan));
        let b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_chan));

        let r_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_high));
        let g_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_high));
        let b_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_high));

        let (mut x_high_low, mut y_high_low, mut z_high_low) = avx2_triple_to_xyz(
            r_high_low,
            g_high_low,
            b_high_low,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            transfer_function,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = avx2_triple_to_lab(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = a;
                z_high_low = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = avx2_triple_to_luv(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = u;
                z_high_low = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = avx_triple_to_lch(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = c;
                z_high_low = h;
            }
        }

        let ptr3 = write_dst_ptr.add(8 * 3 * 2);
        avx_store_and_interleave_v3_direct_f32!(ptr3, x_high_low, y_high_low, z_high_low);

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
            transfer_function,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = avx2_triple_to_lab(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = a;
                z_high_high = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = avx2_triple_to_luv(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = u;
                z_high_high = v;
            }
            XyzTarget::Lch => {
                let (l, u, v) = avx_triple_to_lch(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = u;
                z_high_high = v;
            }
        }

        let ptr4 = write_dst_ptr.add(8 * 3 * 3);
        avx_store_and_interleave_v3_direct_f32!(ptr4, x_high_high, y_high_high, z_high_high);

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
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(a_low))),
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
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(a_high))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx + 8 * 3), a_high_high);
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

        let (mut x_low_low, mut y_low_low, mut z_low_low) = avx2_triple_to_xyz(
            r_low_low,
            g_low_low,
            b_low_low,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            transfer_function,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = avx2_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = avx2_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = avx_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
        }

        let write_dst_ptr = dst_ptr.add(cx * 3);
        avx_store_and_interleave_v3_direct_f32!(write_dst_ptr, x_low_low, y_low_low, z_low_low);

        let r_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r_low));
        let g_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(g_low));
        let b_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(b_low));
        let (mut x_low_high, mut y_low_high, mut z_low_high) = avx2_triple_to_xyz(
            r_low_high,
            g_low_high,
            b_low_high,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            transfer_function,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = avx2_triple_to_lab(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = a;
                z_low_high = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = avx2_triple_to_luv(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = u;
                z_low_high = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = avx_triple_to_lch(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = c;
                z_low_high = h;
            }
        }

        let ptr2 = write_dst_ptr.add(8 * 3);
        avx_store_and_interleave_v3_direct_f32!(ptr2, x_low_high, y_low_high, z_low_high);

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
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(a_low))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx + 8), a_low_high);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave_half::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (mut x_low_low, mut y_low_low, mut z_low_low) = avx2_triple_to_xyz(
            r_low_low,
            g_low_low,
            b_low_low,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            transfer_function,
        );

        match target {
            XyzTarget::Lab => {
                let (l, a, b) = avx2_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::Xyz => {}
            XyzTarget::Luv => {
                let (l, u, v) = avx2_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
            XyzTarget::Lch => {
                let (l, c, h) = avx_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
        }

        let write_dst_ptr = dst_ptr.add(cx * 3);
        avx_store_and_interleave_v3_direct_f32!(write_dst_ptr, x_low_low, y_low_low, z_low_low);

        if USE_ALPHA {
            let a_ptr = (a_linearized as *mut u8).add(a_offset) as *mut f32;

            let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

            let u8_scale = _mm256_set1_ps(1f32 / 255f32);

            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx), a_low_low);
        }

        cx += 8;
    }

    cx
}
