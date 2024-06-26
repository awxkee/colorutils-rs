#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::avx::cie::{avx_lab_to_xyz, avx_lch_to_xyz, avx_luv_to_xyz};
use crate::avx::gamma_curves::get_avx_gamma_transfer;
use crate::avx::{
    _mm256_color_matrix_ps, avx2_deinterleave_rgba_ps, avx2_interleave_rgba_epi8, avx2_pack_s32,
    avx2_pack_u16,
};
use crate::image::ImageConfiguration;
use crate::xyz_target::XyzTarget;
use crate::TransferFunction;

#[inline(always)]
unsafe fn avx_xyza_lab_vld<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: *const f32,
    transfer_function: TransferFunction,
    c1: __m256,
    c2: __m256,
    c3: __m256,
    c4: __m256,
    c5: __m256,
    c6: __m256,
    c7: __m256,
    c8: __m256,
    c9: __m256,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let target: XyzTarget = TARGET.into();
    let transfer = get_avx_gamma_transfer(transfer_function);
    let v_scale_color = _mm256_set1_ps(255f32);
    let pixel_0 = _mm256_loadu_ps(src);
    let pixel_1 = _mm256_loadu_ps(src.add(8));
    let pixel_2 = _mm256_loadu_ps(src.add(16));
    let pixel_3 = _mm256_loadu_ps(src.add(24));
    let (mut r_f32, mut g_f32, mut b_f32, a_f32) =
        avx2_deinterleave_rgba_ps(pixel_0, pixel_1, pixel_2, pixel_3);

    match target {
        XyzTarget::LAB => {
            let (x, y, z) = avx_lab_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::LUV => {
            let (x, y, z) = avx_luv_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::LCH => {
            let (x, y, z) = avx_lch_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        _ => {}
    }

    let (linear_r, linear_g, linear_b) =
        _mm256_color_matrix_ps(r_f32, g_f32, b_f32, c1, c2, c3, c4, c5, c6, c7, c8, c9);

    r_f32 = linear_r;
    g_f32 = linear_g;
    b_f32 = linear_b;

    r_f32 = transfer(r_f32);
    g_f32 = transfer(g_f32);
    b_f32 = transfer(b_f32);
    r_f32 = _mm256_mul_ps(r_f32, v_scale_color);
    g_f32 = _mm256_mul_ps(g_f32, v_scale_color);
    b_f32 = _mm256_mul_ps(b_f32, v_scale_color);
    let a_f32 = _mm256_mul_ps(a_f32, v_scale_color);
    (
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(r_f32)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(g_f32)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(b_f32)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(a_f32)),
    )
}

#[inline(always)]
pub unsafe fn avx_xyza_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
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

    let c1 = _mm256_set1_ps(matrix[0][0]);
    let c2 = _mm256_set1_ps(matrix[0][1]);
    let c3 = _mm256_set1_ps(matrix[0][2]);
    let c4 = _mm256_set1_ps(matrix[1][0]);
    let c5 = _mm256_set1_ps(matrix[1][1]);
    let c6 = _mm256_set1_ps(matrix[1][2]);
    let c7 = _mm256_set1_ps(matrix[2][0]);
    let c8 = _mm256_set1_ps(matrix[2][1]);
    let c9 = _mm256_set1_ps(matrix[2][2]);

    const CHANNELS: usize = 4usize;

    while cx + 32 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset) as *const f32).add(cx * CHANNELS);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = avx_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
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

        let src_ptr_1 = offset_src_ptr.add(8 * CHANNELS);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = avx_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
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

        let src_ptr_2 = offset_src_ptr.add(8 * 2 * CHANNELS);

        let (r_row2_, g_row2_, b_row2_, a_row2_) = avx_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
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

        let src_ptr_3 = offset_src_ptr.add(8 * 3 * CHANNELS);

        let (r_row3_, g_row3_, b_row3_, a_row3_) = avx_xyza_lab_vld::<CHANNELS_CONFIGURATION, TARGET>(
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

        let r_row01 = avx2_pack_s32(r_row0_, r_row1_);
        let g_row01 = avx2_pack_s32(g_row0_, g_row1_);
        let b_row01 = avx2_pack_s32(b_row0_, b_row1_);
        let a_row01 = avx2_pack_s32(a_row0_, a_row1_);

        let r_row23 = avx2_pack_s32(r_row2_, r_row3_);
        let g_row23 = avx2_pack_s32(g_row2_, g_row3_);
        let b_row23 = avx2_pack_s32(b_row2_, b_row3_);
        let a_row23 = avx2_pack_s32(a_row2_, a_row3_);

        let r_row = avx2_pack_u16(r_row01, r_row23);
        let g_row = avx2_pack_u16(g_row01, g_row23);
        let b_row = avx2_pack_u16(b_row01, b_row23);
        let a_row = avx2_pack_u16(a_row01, a_row23);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        let (rgba0, rgba1, rgba2, rgba3) = match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                avx2_interleave_rgba_epi8(r_row, g_row, b_row, a_row)
            }
            ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                avx2_interleave_rgba_epi8(b_row, g_row, r_row, a_row)
            }
        };

        _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba0);
        _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, rgba1);
        _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, rgba2);
        _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, rgba3);

        cx += 32;
    }

    cx
}
