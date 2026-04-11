/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::sse::cie::{sse_lab_to_xyz, sse_lch_to_xyz, sse_luv_to_xyz};
use crate::sse::{_mm_color_matrix_ps, sse_deinterleave_rgb_ps};
use crate::sse::{sse_interleave_ps_rgb, sse_interleave_ps_rgba};
use crate::xyz_target::XyzTarget;
use crate::{store_and_interleave_v3_f32, store_and_interleave_v4_f32};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn sse_xyz_lab_vld<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: *const f32,
    c1: __m128,
    c2: __m128,
    c3: __m128,
    c4: __m128,
    c5: __m128,
    c6: __m128,
    c7: __m128,
    c8: __m128,
    c9: __m128,
) -> (__m128, __m128, __m128) {
    unsafe {
        let target: XyzTarget = TARGET.into();
        let lab_pixel_0 = _mm_loadu_ps(src);
        let lab_pixel_1 = _mm_loadu_ps(src.add(4));
        let lab_pixel_2 = _mm_loadu_ps(src.add(8));
        let (mut r_f32, mut g_f32, mut b_f32) =
            sse_deinterleave_rgb_ps(lab_pixel_0, lab_pixel_1, lab_pixel_2);

        match target {
            XyzTarget::Lab => {
                let (x, y, z) = sse_lab_to_xyz(r_f32, g_f32, b_f32);
                r_f32 = x;
                g_f32 = y;
                b_f32 = z;
            }
            XyzTarget::Luv => {
                let (x, y, z) = sse_luv_to_xyz(r_f32, g_f32, b_f32);
                r_f32 = x;
                g_f32 = y;
                b_f32 = z;
            }
            XyzTarget::Lch => {
                let (x, y, z) = sse_lch_to_xyz(r_f32, g_f32, b_f32);
                r_f32 = x;
                g_f32 = y;
                b_f32 = z;
            }
            _ => {}
        }

        let (linear_r, linear_g, linear_b) =
            _mm_color_matrix_ps(r_f32, g_f32, b_f32, c1, c2, c3, c4, c5, c6, c7, c8, c9);
        (linear_r, linear_g, linear_b)
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn sse_xyz_to_channels<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: &[f32],
    a_channel: &[f32],
    dst: &mut [f32],
    width: u32,
    matrix: &[[f32; 3]; 3],
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.channel_count();

    let mut cx = start_cx;

    let c1 = _mm_set1_ps(matrix[0][0]);
    let c2 = _mm_set1_ps(matrix[0][1]);
    let c3 = _mm_set1_ps(matrix[0][2]);
    let c4 = _mm_set1_ps(matrix[1][0]);
    let c5 = _mm_set1_ps(matrix[1][1]);
    let c6 = _mm_set1_ps(matrix[1][2]);
    let c7 = _mm_set1_ps(matrix[2][0]);
    let c8 = _mm_set1_ps(matrix[2][1]);
    let c9 = _mm_set1_ps(matrix[2][2]);

    let src_channels = 3usize;

    while cx + 4 <= width as usize {
        let offset_src_ptr = unsafe { src.get_unchecked(cx * src_channels..).as_ptr() };

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            sse_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let dst_ptr = unsafe { dst.get_unchecked_mut(cx * channels..).as_mut_ptr() };

        if USE_ALPHA {
            let offset_a_src_ptr = unsafe { a_channel.get_unchecked(cx) };
            let a_row = unsafe { _mm_loadu_ps(offset_a_src_ptr) };

            unsafe {
                store_and_interleave_v4_f32!(
                    dst_ptr,
                    image_configuration,
                    r_row0_,
                    g_row0_,
                    b_row0_,
                    a_row
                );
            }
        } else {
            unsafe {
                store_and_interleave_v3_f32!(
                    dst_ptr,
                    image_configuration,
                    r_row0_,
                    g_row0_,
                    b_row0_
                );
            }
        }

        cx += 4;
    }

    cx
}
