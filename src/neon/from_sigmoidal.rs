/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::neon::sigmoidal::neon_sigmoidal_to_rgb;
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_sigmoidal_vld<const CHANNELS_CONFIGURATION: u8>(
    src: *const f32,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if image_configuration.has_alpha() {
        let sigmoidal_pixel = vld4q_f32(src);
        let (r0, g0, b0) =
            neon_sigmoidal_to_rgb(sigmoidal_pixel.0, sigmoidal_pixel.1, sigmoidal_pixel.2);
        let a0 = vmulq_n_f32(sigmoidal_pixel.3, 255f32);
        return (r0, g0, b0, vcvtaq_u32_f32(a0));
    }
    let sigmoidal_pixel = vld3q_f32(src);
    let (r0, g0, b0) =
        neon_sigmoidal_to_rgb(sigmoidal_pixel.0, sigmoidal_pixel.1, sigmoidal_pixel.2);
    (r0, g0, b0, vdupq_n_u32(0u32))
}

#[inline(always)]
pub unsafe fn neon_from_sigmoidal_row<const CHANNELS_CONFIGURATION: u8>(
    start_cx: usize,
    src: *const f32,
    dst: *mut u8,
    width: u32,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let channels = image_configuration.get_channels_count();

    let mut cx = start_cx;

    while cx + 16 < width as usize {
        let offset_src_ptr = src.add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_sigmoidal_vld::<CHANNELS_CONFIGURATION>(src_ptr_0);

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            neon_sigmoidal_vld::<CHANNELS_CONFIGURATION>(src_ptr_1);

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            neon_sigmoidal_vld::<CHANNELS_CONFIGURATION>(src_ptr_2);

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            neon_sigmoidal_vld::<CHANNELS_CONFIGURATION>(src_ptr_3);

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), vqmovn_u32(r_row1_));
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), vqmovn_u32(g_row1_));
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), vqmovn_u32(b_row1_));

        let r_row23 = vcombine_u16(vqmovn_u32(r_row2_), vqmovn_u32(r_row3_));
        let g_row23 = vcombine_u16(vqmovn_u32(g_row2_), vqmovn_u32(g_row3_));
        let b_row23 = vcombine_u16(vqmovn_u32(b_row2_), vqmovn_u32(b_row3_));

        let r_row = vcombine_u8(vqmovn_u16(r_row01), vqmovn_u16(r_row23));
        let g_row = vcombine_u8(vqmovn_u16(g_row01), vqmovn_u16(g_row23));
        let b_row = vcombine_u8(vqmovn_u16(b_row01), vqmovn_u16(b_row23));

        let dst_ptr = dst.add(cx * channels);

        match image_configuration {
            ImageConfiguration::Rgb => {
                let rgb = uint8x16x3_t(r_row, g_row, b_row);
                vst3q_u8(dst_ptr, rgb);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let a_row01 = vcombine_u16(vqmovn_u32(a_row0_), vqmovn_u32(a_row1_));
                let a_row23 = vcombine_u16(vqmovn_u32(a_row2_), vqmovn_u32(a_row3_));
                let a_row = vcombine_u8(vqmovn_u16(a_row01), vqmovn_u16(a_row23));
                if image_configuration == ImageConfiguration::Rgba {
                    let rgba = uint8x16x4_t(r_row, g_row, b_row, a_row);
                    vst4q_u8(dst_ptr, rgba);
                } else {
                    let bgra = uint8x16x4_t(b_row, g_row, r_row, a_row);
                    vst4q_u8(dst_ptr, bgra);
                }
            }
            ImageConfiguration::Bgr => {
                let bgr = uint8x16x3_t(b_row, g_row, r_row);
                vst3q_u8(dst_ptr, bgr);
            }
        }

        cx += 16;
    }

    cx
}
