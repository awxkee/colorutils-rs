/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use std::arch::aarch64::*;

use erydanos::{vcosq_f32, vsinq_f32};

use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
use crate::neon::math::vcolorq_matrix_f32;
use crate::load_f32_and_deinterleave_direct;

#[inline(always)]
unsafe fn neon_oklab_gamma_vld<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: *const f32,
    m0: float32x4_t,
    m1: float32x4_t,
    m2: float32x4_t,
    m3: float32x4_t,
    m4: float32x4_t,
    m5: float32x4_t,
    m6: float32x4_t,
    m7: float32x4_t,
    m8: float32x4_t,
    c0: float32x4_t,
    c1: float32x4_t,
    c2: float32x4_t,
    c3: float32x4_t,
    c4: float32x4_t,
    c5: float32x4_t,
    c6: float32x4_t,
    c7: float32x4_t,
    c8: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let target: OklabTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let (l, mut a, mut b, a_f32) = load_f32_and_deinterleave_direct!(src, image_configuration);

    if target == OklabTarget::Oklch {
        let a0 = vmulq_f32(a, vcosq_f32(b));
        let b0 = vmulq_f32(a, vsinq_f32(b));
        a = a0;
        b = b0;
    }

    let (mut l_l, mut l_m, mut l_s) =
        vcolorq_matrix_f32(l, a, b, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    l_l = vmulq_f32(vmulq_f32(l_l, l_l), l_l);
    l_m = vmulq_f32(vmulq_f32(l_m, l_m), l_m);
    l_s = vmulq_f32(vmulq_f32(l_s, l_s), l_s);

    let (r_l, g_l, b_l) = vcolorq_matrix_f32(l_l, l_m, l_s, c0, c1, c2, c3, c4, c5, c6, c7, c8);
    (r_l, g_l, b_l, a_f32)
}

#[inline(always)]
pub unsafe fn neon_oklab_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    dst: *mut f32,
    dst_offset: u32,
    width: u32,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        vdupq_n_f32(1f32),
        vdupq_n_f32(0.3963377774f32),
        vdupq_n_f32(0.2158037573f32),
        vdupq_n_f32(1f32),
        vdupq_n_f32(-0.1055613458f32),
        vdupq_n_f32(-0.0638541728f32),
        vdupq_n_f32(1f32),
        vdupq_n_f32(-0.0894841775f32),
        vdupq_n_f32(-1.2914855480f32),
    );

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        vdupq_n_f32(4.0767416621f32),
        vdupq_n_f32(-3.3077115913f32),
        vdupq_n_f32(0.2309699292f32),
        vdupq_n_f32(-1.2684380046f32),
        vdupq_n_f32(2.6097574011f32),
        vdupq_n_f32(-0.3413193965f32),
        vdupq_n_f32(-0.0041960863f32),
        vdupq_n_f32(-0.7034186147f32),
        vdupq_n_f32(1.7076147010f32),
    );

    while cx + 4 < width as usize {
        let v_src_ptr =
            ((src as *mut u8).add(src_offset) as *mut f32).add(cx * channels);

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION, TARGET>(
                v_src_ptr,
                m0,
                m1,
                m2,
                m3,
                m4,
                m5,
                m6,
                m7,
                m8,
                c0,
                c1,
                c2,
                c3,
                c4,
                c5,
                c6,
                c7,
                c8,
            );

        let in_place_ptr =
            ((dst as *mut u8).add(dst_offset as usize) as *mut f32).add(cx * channels);

        if image_configuration.has_alpha() {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(r_row0_, g_row0_, b_row0_, a_row0_)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(b_row0_, g_row0_, r_row0_, a_row0_)
                }
            };
            vst4q_f32(in_place_ptr, store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(r_row0_, g_row0_, b_row0_)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(b_row0_, g_row0_, r_row0_)
                }
            };
            vst3q_f32(in_place_ptr, store_rows);
        }

        cx += 4;
    }

    cx
}
