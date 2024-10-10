/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::neon::cie::{
    neon_triple_to_lab, neon_triple_to_lch, neon_triple_to_luv, neon_triple_to_xyz,
};
use crate::xyz_target::XyzTarget;
use crate::load_f32_and_deinterleave;
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_channels_to_xyza_or_laba<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    matrix: &[[f32; 3]; 3],
) -> usize {
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let cq1 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(0));
    let cq2 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(1));
    let cq3 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(2));
    let cq4 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(0));
    let cq5 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(1));
    let cq6 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(2));
    let cq7 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(0));
    let cq8 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(1));
    let cq9 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(2));

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 4 < width as usize {
        let src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * channels);
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
            XyzTarget::Lch => {
                let (l, c, h) = neon_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
            XyzTarget::Luv => {
                let (l, u, v) = neon_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
        }

        let xyz_low_low = float32x4x4_t(x_low_low, y_low_low, z_low_low, a_chan);
        vst4q_f32(dst_ptr.add(cx * 4), xyz_low_low);

        cx += 4;
    }

    cx
}
