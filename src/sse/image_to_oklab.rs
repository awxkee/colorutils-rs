/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::sse::{sse_deinterleave_rgb_ps, sse_deinterleave_rgba_ps};
use erydanos::{_mm_atan2_ps, _mm_cbrt_fast_ps, _mm_hypot_fast_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
use crate::sse::{
    _mm_color_matrix_ps, sse_interleave_ps_rgb,
    sse_interleave_ps_rgba,
};
use crate::{load_f32_and_deinterleave, store_and_interleave_v3_direct_f32, store_and_interleave_v4_direct_f32};

macro_rules! triple_to_oklab {
    ($r: expr, $g: expr, $b: expr,  $target: expr,
    $c0:expr, $c1:expr, $c2: expr, $c3: expr, $c4:expr, $c5: expr, $c6:expr, $c7: expr, $c8: expr,
        $m0: expr, $m1: expr, $m2: expr, $m3: expr, $m4: expr, $m5: expr, $m6: expr, $m7: expr, $m8: expr
    ) => {{
        let (l_l, l_m, l_s) = _mm_color_matrix_ps(
            $r, $g, $b, $c0, $c1, $c2, $c3, $c4, $c5, $c6, $c7, $c8,
        );

        let l_ = _mm_cbrt_fast_ps(l_l);
        let m_ = _mm_cbrt_fast_ps(l_m);
        let s_ = _mm_cbrt_fast_ps(l_s);

        let (l, mut a, mut b) =
            _mm_color_matrix_ps(l_, m_, s_, $m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, $m8);

        if $target == OklabTarget::Oklch {
            let c = _mm_hypot_fast_ps(a, b);
            let h = _mm_atan2_ps(b, a);
            a = c;
            b = h;
        }

        (l, a, b)
    }};
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_image_to_oklab<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
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
        _mm_set1_ps(0.4122214708f32),
        _mm_set1_ps(0.5363325363f32),
        _mm_set1_ps(0.0514459929f32),
        _mm_set1_ps(0.2119034982f32),
        _mm_set1_ps(0.6806995451f32),
        _mm_set1_ps(0.1073969566f32),
        _mm_set1_ps(0.0883024619f32),
        _mm_set1_ps(0.2817188376f32),
        _mm_set1_ps(0.6299787005f32),
    );

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        _mm_set1_ps(0.2104542553f32),
        _mm_set1_ps(0.7936177850f32),
        _mm_set1_ps(-0.0040720468f32),
        _mm_set1_ps(1.9779984951f32),
        _mm_set1_ps(-2.4285922050f32),
        _mm_set1_ps(0.4505937099f32),
        _mm_set1_ps(0.0259040371f32),
        _mm_set1_ps(0.7827717662f32),
        _mm_set1_ps(-0.8086757660f32),
    );

    while cx + 4 < width as usize {
        let in_place_ptr = dst_ptr.add(cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_f32_and_deinterleave!(in_place_ptr, image_configuration);

        let (l_oklab, a_oklab, b_oklab) = triple_to_oklab!(
            r_chan,
            g_chan,
            b_chan,
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
            store_and_interleave_v4_direct_f32!(in_place_ptr, l_oklab, a_oklab, b_oklab, a_chan);
        } else {
            store_and_interleave_v3_direct_f32!(in_place_ptr, l_oklab, a_oklab, b_oklab);
        }

        cx += 4;
    }

    cx
}
