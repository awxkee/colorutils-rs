/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
use crate::sse::{
    _mm_color_matrix_ps, _mm_cube_ps, perform_sse_gamma_transfer, sse_deinterleave_rgb_ps,
    sse_deinterleave_rgba_ps, sse_interleave_rgb, sse_interleave_rgba,
};
use crate::{
    load_f32_and_deinterleave, store_and_interleave_v3_half_u8, store_and_interleave_v3_u8,
    store_and_interleave_v4_half_u8, store_and_interleave_v4_u8, TransferFunction,
};
use erydanos::{_mm_cos_ps, _mm_sin_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_oklab_vld<const CHANNELS_CONFIGURATION: u8>(
    src: *const f32,
    transfer: TransferFunction,
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
) -> (__m128i, __m128i, __m128i, __m128i) {
    let v_scale_alpha = _mm_set1_ps(255f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let (l, mut a, mut b, mut a_f32) = load_f32_and_deinterleave!(src, image_configuration);

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

    let mut r_f32 = perform_sse_gamma_transfer(transfer, r_l);
    let mut g_f32 = perform_sse_gamma_transfer(transfer, g_l);
    let mut b_f32 = perform_sse_gamma_transfer(transfer, b_l);

    r_f32 = _mm_mul_ps(r_f32, v_scale_alpha);
    g_f32 = _mm_mul_ps(g_f32, v_scale_alpha);
    b_f32 = _mm_mul_ps(b_f32, v_scale_alpha);
    if image_configuration.has_alpha() {
        a_f32 = _mm_mul_ps(a_f32, v_scale_alpha);
    }
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    if image_configuration.has_alpha() {
        (
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(a_f32)),
        )
    } else {
        (
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g_f32)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b_f32)),
            _mm_set1_epi32(255),
        )
    }
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_oklab_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    transfer_function: TransferFunction,
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

    let zeros = _mm_setzero_si128();

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = sse_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_0,
            transfer_function,
            target,
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

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = sse_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_1,
            transfer_function,
            target,
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

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) = sse_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_2,
            transfer_function,
            target,
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

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) = sse_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_3,
            transfer_function,
            target,
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

        let r_row01 = _mm_packus_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packus_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packus_epi32(b_row0_, b_row1_);

        let r_row23 = _mm_packus_epi32(r_row2_, r_row3_);
        let g_row23 = _mm_packus_epi32(g_row2_, g_row3_);
        let b_row23 = _mm_packus_epi32(b_row2_, b_row3_);

        let r_row = _mm_packus_epi16(r_row01, r_row23);
        let g_row = _mm_packus_epi16(g_row01, g_row23);
        let b_row = _mm_packus_epi16(b_row01, b_row23);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if image_configuration.has_alpha() {
            let a_row01 = _mm_packus_epi32(a_row0_, a_row1_);
            let a_row23 = _mm_packus_epi32(a_row2_, a_row3_);
            let a_row = _mm_packus_epi16(a_row01, a_row23);
            store_and_interleave_v4_u8!(dst_ptr, image_configuration, r_row, g_row, b_row, a_row);
        } else {
            store_and_interleave_v3_u8!(dst_ptr, image_configuration, r_row, g_row, b_row);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = sse_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_0,
            transfer_function,
            target,
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

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = sse_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_1,
            transfer_function,
            target,
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

        let r_row01 = _mm_packus_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packus_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packus_epi32(b_row0_, b_row1_);

        let r_row = _mm_packus_epi16(r_row01, zeros);
        let g_row = _mm_packus_epi16(g_row01, zeros);
        let b_row = _mm_packus_epi16(b_row01, zeros);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if image_configuration.has_alpha() {
            let a_row01 = _mm_packus_epi32(a_row0_, a_row1_);
            let a_row = _mm_packus_epi16(a_row01, zeros);
            store_and_interleave_v4_half_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row,
                a_row
            );
        } else {
            store_and_interleave_v3_half_u8!(dst_ptr, image_configuration, r_row, g_row, b_row);
        }

        cx += 8;
    }

    cx
}
