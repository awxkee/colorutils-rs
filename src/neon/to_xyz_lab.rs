/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::load_f32_and_deinterleave;
use crate::neon::cie::{
    neon_triple_to_lab, neon_triple_to_lch, neon_triple_to_luv, neon_triple_to_xyz,
};
use crate::xyz_target::XyzTarget;
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_channels_to_xyz_or_lab<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: &[f32],
    dst: &mut [f32],
    a_linearized: &mut [f32],
    width: u32,
    matrix: &[[f32; 3]; 3],
) -> usize {
    if USE_ALPHA && a_linearized.is_empty() {
        panic!("Null alpha channel with requirements of linearized alpha if not supported");
    }
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.channel_count();
    let mut cx = start_cx;

    let cq1 = vdupq_n_f32(matrix[0][0]);
    let cq2 = vdupq_n_f32(matrix[0][1]);
    let cq3 = vdupq_n_f32(matrix[0][2]);
    let cq4 = vdupq_n_f32(matrix[1][0]);
    let cq5 = vdupq_n_f32(matrix[1][1]);
    let cq6 = vdupq_n_f32(matrix[1][2]);
    let cq7 = vdupq_n_f32(matrix[2][0]);
    let cq8 = vdupq_n_f32(matrix[2][1]);
    let cq9 = vdupq_n_f32(matrix[2][2]);

    while cx + 4 <= width as usize {
        unsafe {
            let src_ptr = src.get_unchecked(cx * channels..).as_ptr();
            let (r_chan, g_chan, b_chan, a_chan) =
                load_f32_and_deinterleave!(src_ptr, image_configuration);

            let (mut x_low_low, mut y_low_low, mut z_low_low) = neon_triple_to_xyz(
                r_chan, g_chan, b_chan, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            );

            match target {
                XyzTarget::Lab => {
                    let (l, a, b) = neon_triple_to_lab(x_low_low, y_low_low, z_low_low);
                    x_low_low = l;
                    y_low_low = a;
                    z_low_low = b;
                }
                XyzTarget::Xyz => {}
                XyzTarget::Luv => {
                    let (l, u, v) = neon_triple_to_luv(x_low_low, y_low_low, z_low_low);
                    x_low_low = l;
                    y_low_low = u;
                    z_low_low = v;
                }
                XyzTarget::Lch => {
                    let (l, c, h) = neon_triple_to_lch(x_low_low, y_low_low, z_low_low);
                    x_low_low = l;
                    y_low_low = c;
                    z_low_low = h;
                }
            }

            let xyz_low_low = float32x4x3_t(x_low_low, y_low_low, z_low_low);
            vst3q_f32(dst.get_unchecked_mut(cx * 3..).as_mut_ptr(), xyz_low_low);

            if USE_ALPHA {
                vst1q_f32(a_linearized.get_unchecked_mut(cx..).as_mut_ptr(), a_chan);
            }
        }

        cx += 4;
    }

    cx
}
