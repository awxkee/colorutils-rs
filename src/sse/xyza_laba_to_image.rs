/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::sse::cie::{sse_lab_to_xyz, sse_lch_to_xyz, sse_luv_to_xyz};
use crate::sse::{
    _mm_color_matrix_ps, perform_sse_gamma_transfer,
    sse_deinterleave_rgba_ps, sse_interleave_rgba,
};
use crate::xyz_target::XyzTarget;
use crate::TransferFunction;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_xyza_lab_vld<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: *const f32,
    transfer_function: TransferFunction,
    c1: __m128,
    c2: __m128,
    c3: __m128,
    c4: __m128,
    c5: __m128,
    c6: __m128,
    c7: __m128,
    c8: __m128,
    c9: __m128,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let target: XyzTarget = TARGET.into();
    let v_scale_color = _mm_set1_ps(255f32);
    let pixel_0 = _mm_loadu_ps(src);
    let pixel_1 = _mm_loadu_ps(src.add(4));
    let pixel_2 = _mm_loadu_ps(src.add(8));
    let pixel_3 = _mm_loadu_ps(src.add(12));
    let (mut r_f32, mut g_f32, mut b_f32, a_f32) =
        sse_deinterleave_rgba_ps(pixel_0, pixel_1, pixel_2, pixel_3);

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

    r_f32 = perform_sse_gamma_transfer(transfer_function, r_f32);
    g_f32 = perform_sse_gamma_transfer(transfer_function, g_f32);
    b_f32 = perform_sse_gamma_transfer(transfer_function, b_f32);
    r_f32 = _mm_mul_ps(r_f32, v_scale_color);
    g_f32 = _mm_mul_ps(g_f32, v_scale_color);
    b_f32 = _mm_mul_ps(b_f32, v_scale_color);
    let a_f32 = _mm_mul_ps(a_f32, v_scale_color);
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    (
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r_f32)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g_f32)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b_f32)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(a_f32)),
    )
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_xyza_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const f32,
    src_offset: usize,
    dst: *mut u8,
    dst_offset: usize,
    width: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

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

    const CHANNELS: usize = 4usize;

    let zeros = _mm_setzero_si128();

    while cx + 16 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * CHANNELS);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = sse_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_0,
            transfer_function,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
        );

        let src_ptr_1 = offset_src_ptr.add(4 * CHANNELS);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = sse_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_1,
            transfer_function,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
        );

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * CHANNELS);

        let (r_row2_, g_row2_, b_row2_, a_row2_) = sse_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_2,
            transfer_function,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
        );

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * CHANNELS);

        let (r_row3_, g_row3_, b_row3_, a_row3_) = sse_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_3,
            transfer_function,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
        );

        let r_row01 = _mm_packs_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packs_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packs_epi32(b_row0_, b_row1_);
        let a_row01 = _mm_packs_epi32(a_row0_, a_row1_);

        let r_row23 = _mm_packs_epi32(r_row2_, r_row3_);
        let g_row23 = _mm_packs_epi32(g_row2_, g_row3_);
        let b_row23 = _mm_packs_epi32(b_row2_, b_row3_);
        let a_row23 = _mm_packs_epi32(a_row2_, a_row3_);

        let r_row = _mm_packus_epi16(r_row01, r_row23);
        let g_row = _mm_packus_epi16(g_row01, g_row23);
        let b_row = _mm_packus_epi16(b_row01, b_row23);
        let a_row = _mm_packus_epi16(a_row01, a_row23);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        let (rgba0, rgba1, rgba2, rgba3) = match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                sse_interleave_rgba(r_row, g_row, b_row, a_row)
            }
            ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                sse_interleave_rgba(b_row, g_row, r_row, a_row)
            }
        };

        _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
        _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
        _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
        _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);

        cx += 16;
    }

    while cx + 8 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * CHANNELS);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = sse_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_0,
            transfer_function,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
        );

        let src_ptr_1 = offset_src_ptr.add(4 * CHANNELS);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = sse_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
            src_ptr_1,
            transfer_function,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
        );

        let r_row01 = _mm_packs_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packs_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packs_epi32(b_row0_, b_row1_);
        let a_row01 = _mm_packs_epi32(a_row0_, a_row1_);

        let r_row = _mm_packus_epi16(r_row01, zeros);
        let g_row = _mm_packus_epi16(g_row01, zeros);
        let b_row = _mm_packus_epi16(b_row01, zeros);
        let a_row = _mm_packus_epi16(a_row01, zeros);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        let (rgba0, rgba1, _, _) = match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                sse_interleave_rgba(r_row, g_row, b_row, a_row)
            }
            ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                sse_interleave_rgba(b_row, g_row, r_row, a_row)
            }
        };

        _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
        _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);

        cx += 8;
    }

    cx
}
