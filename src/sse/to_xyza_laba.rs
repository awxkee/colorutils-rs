/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::load_f32_and_deinterleave;
use crate::sse::cie::{sse_triple_to_lab, sse_triple_to_lch, sse_triple_to_luv, sse_triple_to_xyz};
use crate::sse::*;
use crate::xyz_target::XyzTarget;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_channels_to_xyza_laba<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    matrix: &[[f32; 3]; 3],
) -> usize {
    const CHANNELS: usize = 4;
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    if !image_configuration.has_alpha() {
        panic!("Null alpha channel with requirements of linearized alpha if not supported");
    }
    let mut cx = start_cx;

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

    while cx + 4 < width as usize {
        let src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_f32_and_deinterleave!(src_ptr, image_configuration);

        let (mut x_low_low, mut y_low_low, mut z_low_low) = sse_triple_to_xyz(
            r_chan, g_chan, b_chan, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
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

        let (v0, v1, v2, v3) = sse_interleave_ps_rgba(x_low_low, y_low_low, z_low_low, a_chan);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 12), v3);

        cx += 4;
    }

    cx
}
