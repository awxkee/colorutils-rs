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

use erydanos::{_mm256_cos_ps, _mm256_sin_ps};

use crate::avx::gamma_curves::get_avx_gamma_transfer;
use crate::avx::routines::avx_vld_f32_and_deinterleave_direct;
use crate::avx::{
    _mm256_color_matrix_ps, _mm256_cube_ps, avx2_interleave_rgb, avx2_interleave_rgba_epi8,
    avx2_pack_u16, avx2_pack_u32,
};
use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
use crate::{
    avx_store_and_interleave_v3_half_u8, avx_store_and_interleave_v3_quarter_u8,
    avx_store_and_interleave_v3_u8, avx_store_and_interleave_v4_half_u8,
    avx_store_and_interleave_v4_quarter_u8, avx_store_and_interleave_v4_u8, TransferFunction,
    XYZ_TO_SRGB_D65,
};

#[inline(always)]
unsafe fn avx_oklab_vld<const CHANNELS_CONFIGURATION: u8>(
    src: *const f32,
    transfer: &unsafe fn(__m256) -> __m256,
    oklab_target: OklabTarget,
    m0: __m256,
    m1: __m256,
    m2: __m256,
    m3: __m256,
    m4: __m256,
    m5: __m256,
    m6: __m256,
    m7: __m256,
    m8: __m256,
    c0: __m256,
    c1: __m256,
    c2: __m256,
    c3: __m256,
    c4: __m256,
    c5: __m256,
    c6: __m256,
    c7: __m256,
    c8: __m256,
    x0: __m256,
    x1: __m256,
    x2: __m256,
    x3: __m256,
    x4: __m256,
    x5: __m256,
    x6: __m256,
    x7: __m256,
    x8: __m256,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let v_scale_alpha = _mm256_set1_ps(255f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let (l, mut a, mut b, mut a_f32) =
        avx_vld_f32_and_deinterleave_direct::<CHANNELS_CONFIGURATION>(src);

    if oklab_target == OklabTarget::OKLCH {
        let a0 = _mm256_mul_ps(a, _mm256_cos_ps(b));
        let b0 = _mm256_mul_ps(a, _mm256_sin_ps(b));
        a = a0;
        b = b0;
    }

    let (mut l_l, mut l_m, mut l_s) =
        _mm256_color_matrix_ps(l, a, b, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    l_l = _mm256_cube_ps(l_l);
    l_m = _mm256_cube_ps(l_m);
    l_s = _mm256_cube_ps(l_s);

    let (x, y, z) = _mm256_color_matrix_ps(l_l, l_m, l_s, c0, c1, c2, c3, c4, c5, c6, c7, c8);

    let (r_l, g_l, b_l) = _mm256_color_matrix_ps(x, y, z, x0, x1, x2, x3, x4, x5, x6, x7, x8);

    let mut r_f32 = transfer(r_l);
    let mut g_f32 = transfer(g_l);
    let mut b_f32 = transfer(b_l);

    r_f32 = _mm256_mul_ps(r_f32, v_scale_alpha);
    g_f32 = _mm256_mul_ps(g_f32, v_scale_alpha);
    b_f32 = _mm256_mul_ps(b_f32, v_scale_alpha);
    if image_configuration.has_alpha() {
        a_f32 = _mm256_mul_ps(a_f32, v_scale_alpha);
    }

    if image_configuration.has_alpha() {
        (
            _mm256_cvtps_epi32(_mm256_round_ps::<0x00>(r_f32)),
            _mm256_cvtps_epi32(_mm256_round_ps::<0x00>(g_f32)),
            _mm256_cvtps_epi32(_mm256_round_ps::<0x00>(b_f32)),
            _mm256_cvtps_epi32(_mm256_round_ps::<0x00>(a_f32)),
        )
    } else {
        (
            _mm256_cvtps_epi32(_mm256_round_ps::<0x00>(r_f32)),
            _mm256_cvtps_epi32(_mm256_round_ps::<0x00>(g_f32)),
            _mm256_cvtps_epi32(_mm256_round_ps::<0x00>(b_f32)),
            _mm256_set1_epi32(255),
        )
    }
}

#[inline(always)]
pub unsafe fn avx_oklab_to_image<
    const CHANNELS_CONFIGURATION: u8,
    const TARGET: u8,
    const TRANSFER_FUNCTION: u8,
>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    _: TransferFunction,
) -> usize {
    let transfer_function: TransferFunction = TRANSFER_FUNCTION.into();
    let transfer = get_avx_gamma_transfer(transfer_function);
    let target: OklabTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    // Matrix from XYZ
    let (x0, x1, x2, x3, x4, x5, x6, x7, x8) = (
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(0)),
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(1)),
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(2)),
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(0)),
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(1)),
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(2)),
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(0)),
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(1)),
        _mm256_set1_ps(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(2)),
    );

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        _mm256_set1_ps(1f32),
        _mm256_set1_ps(0.3963377774f32),
        _mm256_set1_ps(0.2158037573f32),
        _mm256_set1_ps(1f32),
        _mm256_set1_ps(-0.1055613458f32),
        _mm256_set1_ps(-0.0638541728f32),
        _mm256_set1_ps(1f32),
        _mm256_set1_ps(-0.0894841775f32),
        _mm256_set1_ps(-1.2914855480f32),
    );

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        _mm256_set1_ps(4.0767416621f32),
        _mm256_set1_ps(-3.3077115913f32),
        _mm256_set1_ps(0.2309699292f32),
        _mm256_set1_ps(-1.2684380046f32),
        _mm256_set1_ps(2.6097574011f32),
        _mm256_set1_ps(-0.3413193965f32),
        _mm256_set1_ps(-0.0041960863f32),
        _mm256_set1_ps(-0.7034186147f32),
        _mm256_set1_ps(1.7076147010f32),
    );

    let zeros = _mm256_setzero_si256();

    while cx + 32 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = avx_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_0, &transfer, target, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4,
            c5, c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
        );

        let src_ptr_1 = offset_src_ptr.add(8 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = avx_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_1, &transfer, target, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4,
            c5, c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
        );

        let src_ptr_2 = offset_src_ptr.add(8 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) = avx_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_2, &transfer, target, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4,
            c5, c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
        );

        let src_ptr_3 = offset_src_ptr.add(8 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) = avx_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_3, &transfer, target, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4,
            c5, c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
        );

        let r_row01 = avx2_pack_u32(r_row0_, r_row1_);
        let g_row01 = avx2_pack_u32(g_row0_, g_row1_);
        let b_row01 = avx2_pack_u32(b_row0_, b_row1_);

        let r_row23 = avx2_pack_u32(r_row2_, r_row3_);
        let g_row23 = avx2_pack_u32(g_row2_, g_row3_);
        let b_row23 = avx2_pack_u32(b_row2_, b_row3_);

        let r_row = avx2_pack_u16(r_row01, r_row23);
        let g_row = avx2_pack_u16(g_row01, g_row23);
        let b_row = avx2_pack_u16(b_row01, b_row23);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if image_configuration.has_alpha() {
            let a_row01 = avx2_pack_u32(a_row0_, a_row1_);
            let a_row23 = avx2_pack_u32(a_row2_, a_row3_);
            let a_row = avx2_pack_u16(a_row01, a_row23);
            avx_store_and_interleave_v4_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row,
                a_row
            );
        } else {
            avx_store_and_interleave_v3_u8!(dst_ptr, image_configuration, r_row, g_row, b_row);
        }

        cx += 32;
    }

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = avx_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_0, &transfer, target, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4,
            c5, c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
        );

        let src_ptr_1 = offset_src_ptr.add(8 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = avx_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_1, &transfer, target, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4,
            c5, c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
        );

        let r_row01 = avx2_pack_u32(r_row0_, r_row1_);
        let g_row01 = avx2_pack_u32(g_row0_, g_row1_);
        let b_row01 = avx2_pack_u32(b_row0_, b_row1_);

        let r_row = avx2_pack_u16(r_row01, zeros);
        let g_row = avx2_pack_u16(g_row01, zeros);
        let b_row = avx2_pack_u16(b_row01, zeros);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if image_configuration.has_alpha() {
            let a_row01 = avx2_pack_u32(a_row0_, a_row1_);
            let a_row = avx2_pack_u16(a_row01, zeros);
            avx_store_and_interleave_v4_half_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row,
                a_row
            );
        } else {
            avx_store_and_interleave_v3_half_u8!(dst_ptr, image_configuration, r_row, g_row, b_row);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = avx_oklab_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_0, &transfer, target, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4,
            c5, c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
        );

        let r_row01 = avx2_pack_u32(r_row0_, zeros);
        let g_row01 = avx2_pack_u32(g_row0_, zeros);
        let b_row01 = avx2_pack_u32(b_row0_, zeros);

        let r_row = avx2_pack_u16(r_row01, zeros);
        let g_row = avx2_pack_u16(g_row01, zeros);
        let b_row = avx2_pack_u16(b_row01, zeros);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if image_configuration.has_alpha() {
            let a_row01 = avx2_pack_u32(a_row0_, zeros);
            let a_row = avx2_pack_u16(a_row01, zeros);
            avx_store_and_interleave_v4_quarter_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row,
                a_row
            );
        } else {
            avx_store_and_interleave_v3_quarter_u8!(
                dst_ptr,
                image_configuration,
                r_row,
                g_row,
                b_row
            );
        }

        cx += 8;
    }

    cx
}
