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

use erydanos::{_mm_atan2_ps, _mm_hypot_fast_ps, _mm_isnan_ps, _mm_mlaf_ps, _mm_pow_ps};

use crate::image::ImageConfiguration;
use crate::image_to_jzazbz::JzazbzTarget;
use crate::sse::{
    _mm_color_matrix_ps, _mm_pow_n_ps, _mm_select_ps, sse_interleave_ps_rgb, sse_interleave_ps_rgba,
};
use crate::sse::{sse_deinterleave_rgb_ps, sse_deinterleave_rgba_ps};
use crate::{
    load_f32_and_deinterleave, store_and_interleave_v3_direct_f32,
    store_and_interleave_v4_direct_f32, SRGB_TO_XYZ_D65,
};

macro_rules! perceptual_quantizer {
    ($color: expr) => {{
        let zeros = _mm_setzero_ps();
        let flush_to_zero_mask = _mm_cmple_ps($color, zeros);
        let xx = _mm_pow_n_ps(_mm_mul_ps($color, _mm_set1_ps(1e-4)), 0.1593017578125);
        let jx = _mm_mlaf_ps(_mm_set1_ps(18.8515625), xx, _mm_set1_ps(0.8359375));
        let den_jx = _mm_mlaf_ps(xx, _mm_set1_ps(18.6875), _mm_set1_ps(1.));
        let rs = _mm_pow_ps(_mm_div_ps(jx, den_jx), _mm_set1_ps(134.034375));
        let flush_nan_to_zero_mask = _mm_isnan_ps(rs);
        _mm_select_ps(
            _mm_or_ps(flush_to_zero_mask, flush_nan_to_zero_mask),
            zeros,
            rs,
        )
    }};
}

macro_rules! triple_to_jzazbz {
    ($r: expr, $g: expr, $b: expr,$target: expr, $luminance: expr
    ) => {{
        let (x0, x1, x2, x3, x4, x5, x6, x7, x8) = (
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(0)),
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(1)),
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(2)),
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(0)),
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(1)),
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(2)),
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(0)),
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(1)),
            _mm_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(2)),
        );

        let (mut x, mut y, mut z) = _mm_color_matrix_ps(
            $r, $g, $b, x0, x1, x2, x3, x4, x5, x6, x7, x8,
        );

        x = _mm_mul_ps(x, $luminance);
        y = _mm_mul_ps(y, $luminance);
        z = _mm_mul_ps(z, $luminance);

        let (l0, l1, l2, l3, l4, l5, l6, l7, l8) = (
            _mm_set1_ps(0.674207838),
            _mm_set1_ps(0.382799340),
            _mm_set1_ps(-0.047570458),
            _mm_set1_ps(0.149284160),
            _mm_set1_ps(0.739628340),
            _mm_set1_ps(0.083327300),
            _mm_set1_ps(0.070941080),
            _mm_set1_ps(0.174768000),
            _mm_set1_ps(0.67097002),
        );

        let (l_l, l_m, l_s) =
            _mm_color_matrix_ps(x, y, z, l0, l1, l2, l3, l4, l5, l6, l7, l8);

        let lp = perceptual_quantizer!(l_l);
        let mp = perceptual_quantizer!(l_m);
        let sp = perceptual_quantizer!(l_s);

        let iz = _mm_mul_ps(_mm_add_ps(lp, mp), _mm_set1_ps(0.5f32));
        let az = _mm_mlaf_ps(
            _mm_set1_ps(3.524000),
            lp,
            _mm_mlaf_ps(_mm_set1_ps(-4.066708), mp, _mm_mul_ps(sp, _mm_set1_ps(0.542708))),
        );
        let bz = _mm_mlaf_ps(
            _mm_set1_ps(0.199076),
            lp,
            _mm_mlaf_ps(_mm_set1_ps(1.096799), mp, _mm_mul_ps(sp, _mm_set1_ps(-1.295875))),
        );
        let num = _mm_mul_ps(iz, _mm_set1_ps(0.44));
        let den = _mm_sub_ps(
            _mm_mlaf_ps(iz, _mm_set1_ps(-0.56), _mm_set1_ps(1.)),
            _mm_set1_ps(1.6295499532821566e-11),
        );
        let jz = _mm_div_ps(num, den);

        match $target {
            JzazbzTarget::Jzazbz => {
                (jz, az, bz)
            }
            JzazbzTarget::Jzczhz => {
                let cz = _mm_hypot_fast_ps(az, bz);
                let hz = _mm_atan2_ps(bz, az);
                (jz, cz, hz)
            }
        }
    }};
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_image_to_jzazbz<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    display_luminance: f32,
) -> usize {
    let target: JzazbzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    let luminance = _mm_set1_ps(display_luminance);

    while cx + 4 < width as usize {
        let src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_f32_and_deinterleave!(src_ptr, image_configuration);

        let (x_low_low, y_low_low, z_low_low) =
            triple_to_jzazbz!(r_chan, g_chan, b_chan, target, luminance);
        if image_configuration.has_alpha() {
            let ptr = dst_ptr.add(cx * 4);
            store_and_interleave_v4_direct_f32!(ptr, x_low_low, y_low_low, z_low_low, a_chan);
        } else {
            let ptr = dst_ptr.add(cx * 3);
            store_and_interleave_v3_direct_f32!(ptr, x_low_low, y_low_low, z_low_low);
        }

        cx += 4;
    }

    cx
}
