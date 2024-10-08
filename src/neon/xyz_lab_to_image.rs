/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::neon::cie::{neon_lab_to_xyz, neon_lch_to_xyz, neon_luv_to_xyz};
use crate::neon::math::*;
use crate::neon::*;
use crate::xyz_target::XyzTarget;
use crate::TransferFunction;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_xyz_lab_vld<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: *const f32,
    transfer_function: TransferFunction,
    c1: float32x4_t,
    c2: float32x4_t,
    c3: float32x4_t,
    c4: float32x4_t,
    c5: float32x4_t,
    c6: float32x4_t,
    c7: float32x4_t,
    c8: float32x4_t,
    c9: float32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t) {
    let target: XyzTarget = TARGET.into();
    let v_scale_color = vdupq_n_f32(255f32);
    let lab_pixel = vld3q_f32(src);
    let (mut r_f32, mut g_f32, mut b_f32) = (lab_pixel.0, lab_pixel.1, lab_pixel.2);

    match target {
        XyzTarget::Lab => {
            let (x, y, z) = neon_lab_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::Luv => {
            let (x, y, z) = neon_luv_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        XyzTarget::Lch => {
            let (x, y, z) = neon_lch_to_xyz(r_f32, g_f32, b_f32);
            r_f32 = x;
            g_f32 = y;
            b_f32 = z;
        }
        _ => {}
    }

    let (linear_r, linear_g, linear_b) =
        vcolorq_matrix_f32(r_f32, g_f32, b_f32, c1, c2, c3, c4, c5, c6, c7, c8, c9);

    r_f32 = linear_r;
    g_f32 = linear_g;
    b_f32 = linear_b;

    r_f32 = neon_perform_gamma_transfer(transfer_function, r_f32);
    g_f32 = neon_perform_gamma_transfer(transfer_function, g_f32);
    b_f32 = neon_perform_gamma_transfer(transfer_function, b_f32);

    r_f32 = vmulq_f32(r_f32, v_scale_color);
    g_f32 = vmulq_f32(g_f32, v_scale_color);
    b_f32 = vmulq_f32(b_f32, v_scale_color);
    (
        vcvtaq_u32_f32(r_f32),
        vcvtaq_u32_f32(g_f32),
        vcvtaq_u32_f32(b_f32),
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub unsafe fn neon_xyz_to_channels<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
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
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut cx = start_cx;

    let c1 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(0));
    let c2 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(1));
    let c3 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(2));
    let c4 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(0));
    let c5 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(1));
    let c6 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(2));
    let c7 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(0));
    let c8 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(1));
    let c9 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(2));

    let src_channels = 3usize;

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset) as *const f32).add(cx * src_channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
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

        let src_ptr_1 = offset_src_ptr.add(4 * src_channels);

        let (r_row1_, g_row1_, b_row1_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
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

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * src_channels);

        let (r_row2_, g_row2_, b_row2_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
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

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * src_channels);

        let (r_row3_, g_row3_, b_row3_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
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

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), vqmovn_u32(r_row1_));
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), vqmovn_u32(g_row1_));
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), vqmovn_u32(b_row1_));

        let r_row23 = vcombine_u16(vqmovn_u32(r_row2_), vqmovn_u32(r_row3_));
        let g_row23 = vcombine_u16(vqmovn_u32(g_row2_), vqmovn_u32(g_row3_));
        let b_row23 = vcombine_u16(vqmovn_u32(b_row2_), vqmovn_u32(b_row3_));

        let r_row = vcombine_u8(vqmovn_u16(r_row01), vqmovn_u16(r_row23));
        let g_row = vcombine_u8(vqmovn_u16(g_row01), vqmovn_u16(g_row23));
        let b_row = vcombine_u8(vqmovn_u16(b_row01), vqmovn_u16(b_row23));

        let dst_ptr = dst.add(dst_offset + cx * channels);

        if USE_ALPHA {
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_low_0_f = vld1q_f32(offset_a_src_ptr);
            let a_row0_ = vcvtaq_u32_f32(vmulq_n_f32(a_low_0_f, 255f32));

            let a_low_1_f = vld1q_f32(offset_a_src_ptr.add(4));
            let a_row1_ = vcvtaq_u32_f32(vmulq_n_f32(a_low_1_f, 255f32));

            let a_low_2_f = vld1q_f32(offset_a_src_ptr.add(8));
            let a_row2_ = vcvtaq_u32_f32(vmulq_n_f32(a_low_2_f, 255f32));

            let a_low_3_f = vld1q_f32(offset_a_src_ptr.add(12));
            let a_row3_ = vcvtaq_u32_f32(vmulq_n_f32(a_low_3_f, 255f32));

            let a_row01 = vcombine_u16(vqmovn_u32(a_row0_), vqmovn_u32(a_row1_));
            let a_row23 = vcombine_u16(vqmovn_u32(a_row2_), vqmovn_u32(a_row3_));
            let a_row = vcombine_u8(vqmovn_u16(a_row01), vqmovn_u16(a_row23));
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x16x4_t(r_row, g_row, b_row, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x16x4_t(b_row, g_row, r_row, a_row)
                }
            };
            vst4q_u8(dst_ptr, store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x16x3_t(r_row, g_row, b_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x16x3_t(b_row, g_row, r_row)
                }
            };
            vst3q_u8(dst_ptr, store_rows);
        }

        cx += 16;
    }

    while cx + 4 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset) as *const f32).add(cx * src_channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
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

        let src_ptr_1 = offset_src_ptr.add(4 * src_channels);

        let (r_row1_, g_row1_, b_row1_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
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

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), vqmovn_u32(r_row1_));
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), vqmovn_u32(g_row1_));
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), vqmovn_u32(b_row1_));

        let r_row = vqmovn_u16(r_row01);
        let g_row = vqmovn_u16(g_row01);
        let b_row = vqmovn_u16(b_row01);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        if USE_ALPHA {
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_low_0_f = vld1q_f32(offset_a_src_ptr);
            let a_row0_ = vcvtaq_u32_f32(vmulq_n_f32(a_low_0_f, 255f32));

            let a_low_1_f = vld1q_f32(offset_a_src_ptr.add(4));
            let a_row1_ = vcvtaq_u32_f32(vmulq_n_f32(a_low_1_f, 255f32));

            let a_row01 = vcombine_u16(vqmovn_u32(a_row0_), vqmovn_u32(a_row1_));
            let a_row = vqmovn_u16(a_row01);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x8x4_t(r_row, g_row, b_row, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x8x4_t(b_row, g_row, r_row, a_row)
                }
            };
            vst4_u8(dst_ptr, store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x8x3_t(r_row, g_row, b_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x8x3_t(b_row, g_row, r_row)
                }
            };
            vst3_u8(dst_ptr, store_rows);
        }

        cx += 8;
    }
    while cx + 8 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset) as *const f32).add(cx * src_channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
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

        let src_ptr_1 = offset_src_ptr.add(4 * src_channels);

        let (r_row1_, g_row1_, b_row1_) =
            neon_xyz_lab_vld::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
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

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), vqmovn_u32(r_row1_));
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), vqmovn_u32(g_row1_));
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), vqmovn_u32(b_row1_));

        let r_row = vqmovn_u16(r_row01);
        let g_row = vqmovn_u16(g_row01);
        let b_row = vqmovn_u16(b_row01);

        let dst_ptr = dst.add(dst_offset + cx * channels);

        if USE_ALPHA {
            let offset_a_src_ptr = ((a_channel as *const u8).add(a_offset) as *const f32).add(cx);
            let a_low_0_f = vld1q_f32(offset_a_src_ptr);
            let a_row0_ = vcvtaq_u32_f32(vmulq_n_f32(a_low_0_f, 255f32));

            let a_low_1_f = vld1q_f32(offset_a_src_ptr.add(4));
            let a_row1_ = vcvtaq_u32_f32(vmulq_n_f32(a_low_1_f, 255f32));

            let a_row01 = vcombine_u16(vqmovn_u32(a_row0_), vqmovn_u32(a_row1_));
            let a_row = vqmovn_u16(a_row01);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x8x4_t(r_row, g_row, b_row, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x8x4_t(b_row, g_row, r_row, a_row)
                }
            };
            let mut transient: [u8; 32] = [0; 32];
            vst4_u8(transient.as_mut_ptr(), store_rows);
            std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr, 4 * 4);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x8x3_t(r_row, g_row, b_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x8x3_t(b_row, g_row, r_row)
                }
            };
            let mut transient: [u8; 24] = [0; 24];
            vst3_u8(transient.as_mut_ptr(), store_rows);
            std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr, 4 * 3);
        }

        cx += 4;
    }

    cx
}
