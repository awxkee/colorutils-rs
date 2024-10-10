/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
use crate::neon::math::vcolorq_matrix_f32;
use crate::load_f32_and_deinterleave;
use erydanos::{vatan2q_f32, vcbrtq_fast_f32, vhypotq_fast_f32};
use std::arch::aarch64::*;

macro_rules! triple_to_oklab {
    ($r: expr, $g: expr, $b: expr, $transfer: expr, $target: expr,
    $c0:expr, $c1:expr, $c2: expr, $c3: expr, $c4:expr, $c5: expr, $c6:expr, $c7: expr, $c8: expr,
        $m0: expr, $m1: expr, $m2: expr, $m3: expr, $m4: expr, $m5: expr, $m6: expr, $m7: expr, $m8: expr
    ) => {{
        let (l_l, l_m, l_s) = vcolorq_matrix_f32(
            $r, $g, $b, $c0, $c1, $c2, $c3, $c4, $c5, $c6, $c7, $c8,
        );

        let l_ = vcbrtq_fast_f32(l_l);
        let m_ = vcbrtq_fast_f32(l_m);
        let s_ = vcbrtq_fast_f32(l_s);

        let (l, mut a, mut b) =
            vcolorq_matrix_f32(l_, m_, s_, $m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, $m8);

        if $target == OklabTarget::Oklch {
            let c = vhypotq_fast_f32(a, b);
            let h = vatan2q_f32(b, a);
            a = c;
            b = h;
        }

        (l, a, b)
    }};
}

#[inline(always)]
pub unsafe fn neon_image_to_oklab<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
) -> usize {
    let target: OklabTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        vdupq_n_f32(0.4122214708f32),
        vdupq_n_f32(0.5363325363f32),
        vdupq_n_f32(0.0514459929f32),
        vdupq_n_f32(0.2119034982f32),
        vdupq_n_f32(0.6806995451f32),
        vdupq_n_f32(0.1073969566f32),
        vdupq_n_f32(0.0883024619f32),
        vdupq_n_f32(0.2817188376f32),
        vdupq_n_f32(0.6299787005f32),
    );

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        vdupq_n_f32(0.2104542553f32),
        vdupq_n_f32(0.7936177850f32),
        vdupq_n_f32(-0.0040720468f32),
        vdupq_n_f32(1.9779984951f32),
        vdupq_n_f32(-2.4285922050f32),
        vdupq_n_f32(0.4505937099f32),
        vdupq_n_f32(0.0259040371f32),
        vdupq_n_f32(0.7827717662f32),
        vdupq_n_f32(-0.8086757660f32),
    );

    while cx + 4 < width as usize {
        let in_place_ptr = dst_ptr.add(cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_f32_and_deinterleave!(in_place_ptr, image_configuration);

        let (x_low_low, y_low_low, z_low_low) = triple_to_oklab!(
            r_chan,
            g_chan,
            b_chan,
            transfer_function,
            target,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8
        );

        if image_configuration.has_alpha() {
            let xyz_low_low = float32x4x4_t(x_low_low, y_low_low, z_low_low, a_chan);
            vst4q_f32(in_place_ptr, xyz_low_low);
        } else {
            let xyz_low_low = float32x4x3_t(x_low_low, y_low_low, z_low_low);
            vst3q_f32(in_place_ptr, xyz_low_low);
        }

        cx += 4;
    }

    cx
}
