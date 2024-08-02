/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::neon::*;
use crate::{load_f32_and_deinterleave, TransferFunction};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_gamma_vld<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: *const f32,
    transfer: &unsafe fn(float32x4_t) -> float32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
    let v_scale_alpha = vdupq_n_f32(255f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let (mut r_f32, mut g_f32, mut b_f32, mut a_f32) =
        load_f32_and_deinterleave!(src, image_configuration);

    r_f32 = transfer(r_f32);
    g_f32 = transfer(g_f32);
    b_f32 = transfer(b_f32);
    r_f32 = vmulq_f32(r_f32, v_scale_alpha);
    g_f32 = vmulq_f32(g_f32, v_scale_alpha);
    b_f32 = vmulq_f32(b_f32, v_scale_alpha);
    if USE_ALPHA {
        a_f32 = vmulq_f32(a_f32, v_scale_alpha);
    }
    (
        vcvtaq_u32_f32(r_f32),
        vcvtaq_u32_f32(g_f32),
        vcvtaq_u32_f32(b_f32),
        vcvtaq_u32_f32(a_f32),
    )
}

#[inline(always)]
pub unsafe fn neon_linear_to_gamma<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
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
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;
    let transfer_function: TransferFunction = TRANSFER_FUNCTION.into();
    let transfer = get_neon_gamma_transfer(transfer_function);

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, &transfer);

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_1, &transfer);

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_2, &transfer);

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_3, &transfer);

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), vqmovn_u32(r_row1_));
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), vqmovn_u32(g_row1_));
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), vqmovn_u32(b_row1_));

        let r_row23 = vcombine_u16(vqmovn_u32(r_row2_), vqmovn_u32(r_row3_));
        let g_row23 = vcombine_u16(vqmovn_u32(g_row2_), vqmovn_u32(g_row3_));
        let b_row23 = vcombine_u16(vqmovn_u32(b_row2_), vqmovn_u32(b_row3_));

        let r_row = vcombine_u8(vqmovn_u16(r_row01), vqmovn_u16(r_row23));
        let g_row = vcombine_u8(vqmovn_u16(g_row01), vqmovn_u16(g_row23));
        let b_row = vcombine_u8(vqmovn_u16(b_row01), vqmovn_u16(b_row23));

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
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

    while cx + 8 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, &transfer);

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_1, &transfer);

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), vqmovn_u32(r_row1_));
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), vqmovn_u32(g_row1_));
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), vqmovn_u32(b_row1_));

        let r_row = vqmovn_u16(r_row01);
        let g_row = vqmovn_u16(g_row01);
        let b_row = vqmovn_u16(b_row01);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
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

    while cx + 4 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, &transfer);

        let zero = vdup_n_u16(0);

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), zero);
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), zero);
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), zero);

        let r_row = vqmovn_u16(r_row01);
        let g_row = vqmovn_u16(g_row01);
        let b_row = vqmovn_u16(b_row01);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
            let a_row01 = vcombine_u16(vqmovn_u32(a_row0_), zero);
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
