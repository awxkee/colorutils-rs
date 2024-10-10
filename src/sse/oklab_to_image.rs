/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
use crate::sse::{
    _mm_color_matrix_ps, _mm_cube_ps, sse_deinterleave_rgb_ps, sse_deinterleave_rgba_ps,
};
use crate::sse::{sse_interleave_ps_rgb, sse_interleave_ps_rgba};
use crate::{load_f32_and_deinterleave, store_and_interleave_v3_f32, store_and_interleave_v4_f32};
use erydanos::{_mm_cos_ps, _mm_sin_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_oklab_vld<const CHANNELS_CONFIGURATION: u8>(
    src: *const f32,
    oklab_target: OklabTarget,
    m0: __m128,
    m1: __m128,
    m2: __m128,
    m3: __m128,
    m4: __m128,
    m5: __m128,
    m6: __m128,
    m7: __m128,
    m8: __m128,
    c0: __m128,
    c1: __m128,
    c2: __m128,
    c3: __m128,
    c4: __m128,
    c5: __m128,
    c6: __m128,
    c7: __m128,
    c8: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let (l, mut a, mut b, a_f32) = load_f32_and_deinterleave!(src, image_configuration);

    if oklab_target == OklabTarget::Oklch {
        let a0 = _mm_mul_ps(a, _mm_cos_ps(b));
        let b0 = _mm_mul_ps(a, _mm_sin_ps(b));
        a = a0;
        b = b0;
    }

    let (mut l_l, mut l_m, mut l_s) =
        _mm_color_matrix_ps(l, a, b, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    l_l = _mm_cube_ps(l_l);
    l_m = _mm_cube_ps(l_m);
    l_s = _mm_cube_ps(l_s);

    let (r_l, g_l, b_l) = _mm_color_matrix_ps(l_l, l_m, l_s, c0, c1, c2, c3, c4, c5, c6, c7, c8);
    (r_l, g_l, b_l, a_f32)
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_oklab_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    dst: *mut f32,
    dst_offset: u32,
    width: u32,
) -> usize {
    let target: OklabTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        _mm_set1_ps(1f32),
        _mm_set1_ps(0.3963377774f32),
        _mm_set1_ps(0.2158037573f32),
        _mm_set1_ps(1f32),
        _mm_set1_ps(-0.1055613458f32),
        _mm_set1_ps(-0.0638541728f32),
        _mm_set1_ps(1f32),
        _mm_set1_ps(-0.0894841775f32),
        _mm_set1_ps(-1.2914855480f32),
    );

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        _mm_set1_ps(4.0767416621f32),
        _mm_set1_ps(-3.3077115913f32),
        _mm_set1_ps(0.2309699292f32),
        _mm_set1_ps(-1.2684380046f32),
        _mm_set1_ps(2.6097574011f32),
        _mm_set1_ps(-0.3413193965f32),
        _mm_set1_ps(-0.0041960863f32),
        _mm_set1_ps(-0.7034186147f32),
        _mm_set1_ps(1.7076147010f32),
    );

    while cx + 4 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = sse_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_0, target, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4, c5, c6, c7,
            c8,
        );

        let dst_ptr = ((dst as *mut u8).add(dst_offset as usize) as *mut f32).add(cx * channels);

        if image_configuration.has_alpha() {
            store_and_interleave_v4_f32!(
                dst_ptr,
                image_configuration,
                r_row0_,
                g_row0_,
                b_row0_,
                a_row0_
            );
        } else {
            store_and_interleave_v3_f32!(dst_ptr, image_configuration, r_row0_, g_row0_, b_row0_);
        }

        cx += 4;
    }

    cx
}
