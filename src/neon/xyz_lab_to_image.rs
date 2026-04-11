/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::neon::cie::{neon_lab_to_xyz, neon_lch_to_xyz, neon_luv_to_xyz};
use crate::neon::math::*;
use crate::xyz_target::XyzTarget;
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn neon_xyz_lab_vld<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: *const f32,
    c1: float32x4_t,
    c2: float32x4_t,
    c3: float32x4_t,
    c4: float32x4_t,
    c5: float32x4_t,
    c6: float32x4_t,
    c7: float32x4_t,
    c8: float32x4_t,
    c9: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    unsafe {
        let target: XyzTarget = TARGET.into();
        let lab_pixel = vld3q_f32(src);
        let (mut r_f32, mut g_f32, mut b_f32) = (lab_pixel.0, lab_pixel.1, lab_pixel.2);

        match target {
            XyzTarget::Lab => {
                let (x, y, z) = neon_lab_to_xyz(r_f32, g_f32, b_f32);
                r_f32 = x;
                g_f32 = y;
                b_f32 = z;
            }
            XyzTarget::Luv => {
                let (x, y, z) = neon_luv_to_xyz(r_f32, g_f32, b_f32);
                r_f32 = x;
                g_f32 = y;
                b_f32 = z;
            }
            XyzTarget::Lch => {
                let (x, y, z) = neon_lch_to_xyz(r_f32, g_f32, b_f32);
                r_f32 = x;
                g_f32 = y;
                b_f32 = z;
            }
            _ => {}
        }

        let (linear_r, linear_g, linear_b) =
            vcolorq_matrix_f32(r_f32, g_f32, b_f32, c1, c2, c3, c4, c5, c6, c7, c8, c9);

        (linear_r, linear_g, linear_b)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
pub(crate) fn neon_xyz_to_channels<
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

    let c1 = vdupq_n_f32(matrix[0][0]);
    let c2 = vdupq_n_f32(matrix[0][1]);
    let c3 = vdupq_n_f32(matrix[0][2]);
    let c4 = vdupq_n_f32(matrix[1][0]);
    let c5 = vdupq_n_f32(matrix[1][1]);
    let c6 = vdupq_n_f32(matrix[1][2]);
    let c7 = vdupq_n_f32(matrix[2][0]);
    let c8 = vdupq_n_f32(matrix[2][1]);
    let c9 = vdupq_n_f32(matrix[2][2]);

    let src_channels = 3usize;

    while cx + 4 <= width as usize {
        let offset_src_ptr = unsafe { src.get_unchecked(cx * src_channels..).as_ptr() };

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let dst_ptr = unsafe { dst.get_unchecked_mut(cx * channels..).as_mut_ptr() };

        if USE_ALPHA {
            let offset_a_src_ptr = unsafe { a_channel.get_unchecked(cx..).as_ptr() };
            let a_row = unsafe { vld1q_f32(offset_a_src_ptr) };
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(r_row0_, g_row0_, b_row0_, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(b_row0_, g_row0_, r_row0_, a_row)
                }
            };
            unsafe {
                vst4q_f32(dst_ptr, store_rows);
            }
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(r_row0_, g_row0_, b_row0_)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(b_row0_, g_row0_, r_row0_)
                }
            };
            unsafe {
                vst3q_f32(dst_ptr, store_rows);
            }
        }

        cx += 4;
    }

    cx
}
