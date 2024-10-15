/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::neon::cie::{neon_lab_to_xyz, neon_lch_to_xyz, neon_luv_to_xyz};
use crate::neon::math::vcolorq_matrix_f32;
use crate::xyz_target::XyzTarget;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_xyza_lab_vld<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
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
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let target: XyzTarget = TARGET.into();
    let lab_pixel = vld4q_f32(src);
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

    (linear_r, linear_g, linear_b, lab_pixel.3)
}

#[inline(always)]
pub unsafe fn neon_xyza_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    dst: *mut f32,
    dst_offset: usize,
    width: u32,
    matrix: &[[f32; 3]; 3],
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut cx = start_cx;

    let c1 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(0));
    let c2 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(1));
    let c3 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(2));
    let c4 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(0));
    let c5 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(1));
    let c6 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(2));
    let c7 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(0));
    let c8 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(1));
    let c9 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(2));

    const CHANNELS: usize = 4usize;

    while cx + 4 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * CHANNELS);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
                src_ptr_0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let dst_ptr = ((dst as *mut u8).add(dst_offset) as *mut f32).add(cx * channels);

        let store_rows = match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                float32x4x4_t(r_row0_, g_row0_, b_row0_, a_row0_)
            }
            ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                float32x4x4_t(b_row0_, g_row0_, r_row0_, a_row0_)
            }
        };
        vst4q_f32(dst_ptr, store_rows);

        cx += 4;
    }

    cx
}
