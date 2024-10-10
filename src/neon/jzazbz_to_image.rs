/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use erydanos::{vcosq_f32, visnanq_f32, vmlafq_f32, vpowq_f32, vsinq_f32};

use crate::image::ImageConfiguration;
use crate::image_to_jzazbz::JzazbzTarget;
use crate::neon::math::{vcolorq_matrix_f32, vpowq_n_f32};
use crate::{load_f32_and_deinterleave_direct, XYZ_TO_SRGB_D65};

macro_rules! perceptual_quantizer_inverse {
    ($color: expr) => {{
        let flush_to_zero_mask = vclezq_f32($color);
        let xx = vpowq_n_f32($color, 7.460772656268214e-03);
        let num = vsubq_f32(vdupq_n_f32(0.8359375), xx);
        let den = vmlafq_f32(xx, vdupq_n_f32(18.6875), vdupq_n_f32(-18.8515625));
        let rs = vmulq_n_f32(
            vpowq_f32(vdivq_f32(num, den), vdupq_n_f32(6.277394636015326)),
            1e4,
        );
        let flush_nan_mask = visnanq_f32(rs);
        vbslq_f32(
            vorrq_u32(flush_to_zero_mask, flush_nan_mask),
            vdupq_n_f32(0.),
            rs,
        )
    }};
}

#[inline(always)]
unsafe fn neon_jzazbz_gamma_vld<const CHANNELS_CONFIGURATION: u8>(
    src: *const f32,
    target: JzazbzTarget,
    luminance: f32,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let (jz, mut az, mut bz, a_f32) = load_f32_and_deinterleave_direct!(src, image_configuration);

    if target == JzazbzTarget::Jzczhz {
        let cz = az;
        let hz = bz;
        az = vmulq_f32(cz, vcosq_f32(hz));
        bz = vmulq_f32(cz, vsinq_f32(hz));
    }

    let jz = vaddq_f32(jz, vdupq_n_f32(1.6295499532821566e-11));
    let iz = vdivq_f32(
        jz,
        vmlafq_f32(jz, vdupq_n_f32(0.56f32), vdupq_n_f32(0.44f32)),
    );

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        vdupq_n_f32(1f32),
        vdupq_n_f32(1.386050432715393e-1),
        vdupq_n_f32(5.804731615611869e-2),
        vdupq_n_f32(1f32),
        vdupq_n_f32(-1.386050432715393e-1),
        vdupq_n_f32(-5.804731615611891e-2),
        vdupq_n_f32(1f32),
        vdupq_n_f32(-9.601924202631895e-2),
        vdupq_n_f32(-8.118918960560390e-1),
    );

    let (mut l_l, mut l_m, mut l_s) =
        vcolorq_matrix_f32(iz, az, bz, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    l_l = perceptual_quantizer_inverse!(l_l);
    l_m = perceptual_quantizer_inverse!(l_m);
    l_s = perceptual_quantizer_inverse!(l_s);

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        vdupq_n_f32(1.661373055774069e+00),
        vdupq_n_f32(-9.145230923250668e-01),
        vdupq_n_f32(2.313620767186147e-01),
        vdupq_n_f32(-3.250758740427037e-01),
        vdupq_n_f32(1.571847038366936e+00),
        vdupq_n_f32(-2.182538318672940e-01),
        vdupq_n_f32(-9.098281098284756e-02),
        vdupq_n_f32(-3.127282905230740e-01),
        vdupq_n_f32(1.522766561305260e+00),
    );

    let (mut x, mut y, mut z) =
        vcolorq_matrix_f32(l_l, l_m, l_s, c0, c1, c2, c3, c4, c5, c6, c7, c8);

    x = vmulq_n_f32(x, luminance);
    y = vmulq_n_f32(y, luminance);
    z = vmulq_n_f32(z, luminance);

    let (x0, x1, x2, x3, x4, x5, x6, x7, x8) = (
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(0)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(1)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(2)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(0)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(1)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(2)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(0)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(1)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(2)),
    );

    let (r_l, g_l, b_l) = vcolorq_matrix_f32(x, y, z, x0, x1, x2, x3, x4, x5, x6, x7, x8);

    (r_l, g_l, b_l, a_f32)
}

pub unsafe fn neon_jzazbz_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut f32,
    dst_offset: u32,
    width: u32,
    display_luminance: f32,
) -> usize {
    let target: JzazbzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let luminance_scale: f32 = 1. / display_luminance;

    while cx + 4 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_jzazbz_gamma_vld::<CHANNELS_CONFIGURATION>(src_ptr_0, target, luminance_scale);

        let dst_ptr = ((dst as *mut u8).add(dst_offset as usize) as *mut f32).add(cx * channels);

        if image_configuration.has_alpha() {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(r_row0_, g_row0_, b_row0_, a_row0_)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(b_row0_, g_row0_, r_row0_, a_row0_)
                }
            };
            vst4q_f32(dst_ptr, store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(r_row0_, g_row0_, b_row0_)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(b_row0_, g_row0_, r_row0_)
                }
            };
            vst3q_f32(dst_ptr, store_rows);
        }

        cx += 4;
    }

    cx
}
