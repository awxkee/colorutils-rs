/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::sse::cie::{sse_triple_to_lab, sse_triple_to_lch, sse_triple_to_luv, sse_triple_to_xyz};
use crate::sse::*;
use crate::xyz_target::XyzTarget;
use crate::load_f32_and_deinterleave;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_channels_to_xyz_or_lab<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    a_linearized: *mut f32,
    a_offset: usize,
    matrix: &[[f32; 3]; 3],
) -> usize {
    if USE_ALPHA && a_linearized.is_null() {
        panic!("Null alpha channel with requirements of linearized alpha if not supported");
    }
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
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
        let src_ptr = ((src as * const u8).add(src_offset) as *const f32).add(cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_f32_and_deinterleave!(src_ptr, image_configuration);

        let (mut x_low_low, mut y_low_low, mut z_low_low) = sse_triple_to_xyz(
            r_chan,
            g_chan,
            b_chan,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
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

        let (v0, v1, v2) = sse_interleave_ps_rgb(x_low_low, y_low_low, z_low_low);
        _mm_storeu_ps(dst_ptr.add(cx * 3), v0);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 8), v2);

        if USE_ALPHA {
            let a_ptr = (a_linearized as *mut u8).add(a_offset) as *mut f32;

            _mm_storeu_ps(a_ptr.add(cx), a_chan);
        }

        cx += 4;
    }

    cx
}
