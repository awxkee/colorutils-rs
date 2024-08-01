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

use erydanos::{_mm_cos_ps, _mm_isnan_ps, _mm_mlaf_ps, _mm_pow_ps, _mm_sin_ps};

use crate::image::ImageConfiguration;
use crate::image_to_jzazbz::JzazbzTarget;
use crate::sse::{
    _mm_color_matrix_ps, _mm_pow_n_ps, _mm_select_ps, get_sse_gamma_transfer,
    sse_deinterleave_rgb_ps, sse_deinterleave_rgba_ps, sse_interleave_rgb, sse_interleave_rgba,
};
use crate::{
    load_f32_and_deinterleave_direct, store_and_interleave_v3_u8, store_and_interleave_v4_u8,
    TransferFunction, XYZ_TO_SRGB_D65,
};

macro_rules! perceptual_quantizer_inverse {
    ($color: expr) => {{
        let zeros = _mm_setzero_ps();
        let flush_to_zero_mask = _mm_cmple_ps($color, zeros);
        let xx = _mm_pow_n_ps($color, 7.460772656268214e-03);
        let num = _mm_sub_ps(_mm_set1_ps(0.8359375), xx);
        let den = _mm_mlaf_ps(xx, _mm_set1_ps(18.6875), _mm_set1_ps(-18.8515625));
        let rs = _mm_mul_ps(
            _mm_pow_ps(_mm_div_ps(num, den), _mm_set1_ps(6.277394636015326)),
            _mm_set1_ps(1e4),
        );
        let flush_nan_to_zero_mask = _mm_isnan_ps(rs);
        _mm_select_ps(
            _mm_or_ps(flush_to_zero_mask, flush_nan_to_zero_mask),
            zeros,
            rs,
        )
    }};
}

#[inline(always)]
unsafe fn sse_jzazbz_vld<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: *const f32,
    transfer_function: TransferFunction,
    luminance_scale: __m128,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let target: JzazbzTarget = TARGET.into();
    let transfer = get_sse_gamma_transfer(transfer_function);
    let v_scale_alpha = _mm_set1_ps(255f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let (jz, mut az, mut bz, mut a_f32) =
        load_f32_and_deinterleave_direct!(src, image_configuration);

    if target == JzazbzTarget::JZCZHZ {
        let cz = az;
        let hz = bz;
        az = _mm_mul_ps(cz, _mm_cos_ps(hz));
        bz = _mm_mul_ps(cz, _mm_sin_ps(hz));
    }

    let jz = _mm_add_ps(jz, _mm_set1_ps(1.6295499532821566e-11));
    let iz = _mm_div_ps(
        jz,
        _mm_mlaf_ps(jz, _mm_set1_ps(0.56f32), _mm_set1_ps(0.44f32)),
    );

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        _mm_set1_ps(1f32),
        _mm_set1_ps(1.386050432715393e-1),
        _mm_set1_ps(5.804731615611869e-2),
        _mm_set1_ps(1f32),
        _mm_set1_ps(-1.386050432715393e-1),
        _mm_set1_ps(-5.804731615611891e-2),
        _mm_set1_ps(1f32),
        _mm_set1_ps(-9.601924202631895e-2),
        _mm_set1_ps(-8.118918960560390e-1),
    );

    let (mut l_l, mut l_m, mut l_s) =
        _mm_color_matrix_ps(iz, az, bz, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    l_l = perceptual_quantizer_inverse!(l_l);
    l_m = perceptual_quantizer_inverse!(l_m);
    l_s = perceptual_quantizer_inverse!(l_s);

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        _mm_set1_ps(1.661373055774069e+00),
        _mm_set1_ps(-9.145230923250668e-01),
        _mm_set1_ps(2.313620767186147e-01),
        _mm_set1_ps(-3.250758740427037e-01),
        _mm_set1_ps(1.571847038366936e+00),
        _mm_set1_ps(-2.182538318672940e-01),
        _mm_set1_ps(-9.098281098284756e-02),
        _mm_set1_ps(-3.127282905230740e-01),
        _mm_set1_ps(1.522766561305260e+00),
    );

    let (mut x, mut y, mut z) =
        _mm_color_matrix_ps(l_l, l_m, l_s, c0, c1, c2, c3, c4, c5, c6, c7, c8);

    x = _mm_mul_ps(x, luminance_scale);
    y = _mm_mul_ps(y, luminance_scale);
    z = _mm_mul_ps(z, luminance_scale);

    let (x0, x1, x2, x3, x4, x5, x6, x7, x8) = (
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(0)),
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(1)),
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(2)),
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(0)),
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(1)),
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(2)),
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(0)),
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(1)),
        _mm_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(2)),
    );

    let (r_l, g_l, b_l) = _mm_color_matrix_ps(x, y, z, x0, x1, x2, x3, x4, x5, x6, x7, x8);

    let mut r_f32 = transfer(r_l);
    let mut g_f32 = transfer(g_l);
    let mut b_f32 = transfer(b_l);
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

#[inline(always)]
pub unsafe fn sse_jzazbz_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let luminance_scale = _mm_set1_ps(1. / display_luminance);

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = sse_jzazbz_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_0,
            transfer_function,
            luminance_scale,
        );

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = sse_jzazbz_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_1,
            transfer_function,
            luminance_scale,
        );

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) = sse_jzazbz_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_2,
            transfer_function,
            luminance_scale,
        );

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) = sse_jzazbz_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_3,
            transfer_function,
            luminance_scale,
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

    cx
}
