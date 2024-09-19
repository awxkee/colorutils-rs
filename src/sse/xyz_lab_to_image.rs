/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::sse::cie::{sse_lab_to_xyz, sse_lch_to_xyz, sse_luv_to_xyz};
use crate::sse::{
    _mm_color_matrix_ps, get_sse_gamma_transfer, sse_deinterleave_rgb_ps, sse_interleave_rgb,
    sse_interleave_rgba,
};
use crate::xyz_target::XyzTarget;
use crate::TransferFunction;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_xyz_lab_vld<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: *const f32,
    transfer: &unsafe fn(__m128) -> __m128,
    c1: __m128,
    c2: __m128,
    c3: __m128,
    c4: __m128,
    c5: __m128,
    c6: __m128,
    c7: __m128,
    c8: __m128,
    c9: __m128,
) -> (__m128i, __m128i, __m128i) {
    let target: XyzTarget = TARGET.into();
    let v_scale_color = _mm_set1_ps(255f32);
    let lab_pixel_0 = _mm_loadu_ps(src);
    let lab_pixel_1 = _mm_loadu_ps(src.add(4));
    let lab_pixel_2 = _mm_loadu_ps(src.add(8));
    let (mut r_f32, mut g_f32, mut b_f32) =
        sse_deinterleave_rgb_ps(lab_pixel_0, lab_pixel_1, lab_pixel_2);

    match target {
        XyzTarget::Lab => {
            let (x, y, z) = sse_lab_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::Luv => {
            let (x, y, z) = sse_luv_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::Lch => {
            let (x, y, z) = sse_lch_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        _ => {}
    }

    let (linear_r, linear_g, linear_b) =
        _mm_color_matrix_ps(r_f32, g_f32, b_f32, c1, c2, c3, c4, c5, c6, c7, c8, c9);

    r_f32 = linear_r;
    g_f32 = linear_g;
    b_f32 = linear_b;

    r_f32 = transfer(r_f32);
    g_f32 = transfer(g_f32);
    b_f32 = transfer(b_f32);
    r_f32 = _mm_mul_ps(r_f32, v_scale_color);
    g_f32 = _mm_mul_ps(g_f32, v_scale_color);
    b_f32 = _mm_mul_ps(b_f32, v_scale_color);
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    (
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r_f32)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g_f32)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b_f32)),
    )
}

#[inline(always)]
pub unsafe fn sse_xyz_to_channels<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
    const TRANSFER_FUNCTION: u8,
>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    a_channel: *const f32,
    a_offset: usize,
    dst: *mut u8,
    dst_offset: usize,
    width: u32,
    matrix: &[[f32; 3]; 3],
    _: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Alpha may be set only on images with alpha");
        }
    }

    let transfer_function: TransferFunction = TRANSFER_FUNCTION.into();
    let transfer = get_sse_gamma_transfer(transfer_function);

    let channels = image_configuration.get_channels_count();

    let mut cx = start_cx;

    let c1 = _mm_set1_ps(*matrix.get_unchecked(0).get_unchecked(0));
    let c2 = _mm_set1_ps(*matrix.get_unchecked(0).get_unchecked(1));
    let c3 = _mm_set1_ps(*matrix.get_unchecked(0).get_unchecked(2));
    let c4 = _mm_set1_ps(*matrix.get_unchecked(1).get_unchecked(0));
    let c5 = _mm_set1_ps(*matrix.get_unchecked(1).get_unchecked(1));
    let c6 = _mm_set1_ps(*matrix.get_unchecked(1).get_unchecked(2));
    let c7 = _mm_set1_ps(*matrix.get_unchecked(2).get_unchecked(0));
    let c8 = _mm_set1_ps(*matrix.get_unchecked(2).get_unchecked(1));
    let c9 = _mm_set1_ps(*matrix.get_unchecked(2).get_unchecked(2));

    let src_channels = 3usize;

    let color_rescale = _mm_set1_ps(255f32);

    let zeros = _mm_setzero_si128();

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset) as *const f32).add(cx * src_channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            sse_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let src_ptr_1 = offset_src_ptr.add(4 * src_channels);

        let (r_row1_, g_row1_, b_row1_) =
            sse_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_1, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * src_channels);

        let (r_row2_, g_row2_, b_row2_) =
            sse_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_2, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * src_channels);

        let (r_row3_, g_row3_, b_row3_) =
            sse_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_3, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let r_row01 = _mm_packs_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packs_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packs_epi32(b_row0_, b_row1_);

        let r_row23 = _mm_packs_epi32(r_row2_, r_row3_);
        let g_row23 = _mm_packs_epi32(g_row2_, g_row3_);
        let b_row23 = _mm_packs_epi32(b_row2_, b_row3_);

        let r_row = _mm_packus_epi16(r_row01, r_row23);
        let g_row = _mm_packus_epi16(g_row01, g_row23);
        let b_row = _mm_packus_epi16(b_row01, b_row23);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        if USE_ALPHA {
            const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_low_0_f = _mm_loadu_ps(offset_a_src_ptr);
            let a_row0_ = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                a_low_0_f,
                color_rescale,
            )));

            let a_low_1_f = _mm_loadu_ps(offset_a_src_ptr.add(4));
            let a_row1_ = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                a_low_1_f,
                color_rescale,
            )));

            let a_low_2_f = _mm_loadu_ps(offset_a_src_ptr.add(8));
            let a_row2_ = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                a_low_2_f,
                color_rescale,
            )));

            let a_low_3_f = _mm_loadu_ps(offset_a_src_ptr.add(12));
            let a_row3_ = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                a_low_3_f,
                color_rescale,
            )));

            let a_row01 = _mm_packs_epi32(a_row0_, a_row1_);
            let a_row23 = _mm_packs_epi32(a_row2_, a_row3_);
            let a_row = _mm_packus_epi16(a_row01, a_row23);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    sse_interleave_rgba(r_row, g_row, b_row, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    sse_interleave_rgba(b_row, g_row, r_row, a_row)
                }
            };
            _mm_storeu_si128(dst_ptr as *mut __m128i, store_rows.0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, store_rows.1);
            _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, store_rows.2);
            _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, store_rows.3);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    sse_interleave_rgb(r_row, g_row, b_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    sse_interleave_rgb(b_row, g_row, r_row)
                }
            };
            _mm_storeu_si128(dst_ptr as *mut __m128i, store_rows.0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, store_rows.1);
            _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, store_rows.2);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset) as *const f32).add(cx * src_channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            sse_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_0, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let src_ptr_1 = offset_src_ptr.add(4 * src_channels);

        let (r_row1_, g_row1_, b_row1_) =
            sse_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                src_ptr_1, &transfer, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            );

        let r_row01 = _mm_packs_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packs_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packs_epi32(b_row0_, b_row1_);

        let r_row = _mm_packus_epi16(r_row01, zeros);
        let g_row = _mm_packus_epi16(g_row01, zeros);
        let b_row = _mm_packus_epi16(b_row01, zeros);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        if USE_ALPHA {
            const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_low_0_f = _mm_loadu_ps(offset_a_src_ptr);
            let a_row0_ = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                a_low_0_f,
                color_rescale,
            )));

            let a_low_1_f = _mm_loadu_ps(offset_a_src_ptr.add(4));
            let a_row1_ = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
                a_low_1_f,
                color_rescale,
            )));

            let a_row01 = _mm_packs_epi32(a_row0_, a_row1_);
            let a_row = _mm_packus_epi16(a_row01, zeros);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    sse_interleave_rgba(r_row, g_row, b_row, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    sse_interleave_rgba(b_row, g_row, r_row, a_row)
                }
            };
            _mm_storeu_si128(dst_ptr as *mut __m128i, store_rows.0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, store_rows.1);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    sse_interleave_rgb(r_row, g_row, b_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    sse_interleave_rgb(b_row, g_row, r_row)
                }
            };
            _mm_storeu_si128(dst_ptr as *mut __m128i, store_rows.0);
            let regi = store_rows.1;
            std::ptr::copy_nonoverlapping(&regi as *const _ as *const u8, dst_ptr.add(16), 8);
        }

        cx += 8;
    }

    cx
}
