/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::neon::*;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_triple_to_linear(
    r: uint32x4_t,
    g: uint32x4_t,
    b: uint32x4_t,
    transfer: &unsafe fn(float32x4_t) -> float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let r_f = vmulq_n_f32(vcvtq_f32_u32(r), 1f32 / 255f32);
    let g_f = vmulq_n_f32(vcvtq_f32_u32(g), 1f32 / 255f32);
    let b_f = vmulq_n_f32(vcvtq_f32_u32(b), 1f32 / 255f32);
    let r_linear = transfer(r_f);
    let g_linear = transfer(g_f);
    let b_linear = transfer(b_f);
    (r_linear, g_linear, b_linear)
}

#[inline(always)]
pub unsafe fn neon_channels_to_linear<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer = get_neon_linear_transfer(transfer_function);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 16 < width as usize {
        let (r_chan, g_chan, b_chan, a_chan);
        let src_ptr = src.add(src_offset + cx * channels);
        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let ldr = vld3q_u8(src_ptr);
                if image_configuration == ImageConfiguration::Rgb {
                    r_chan = ldr.0;
                    g_chan = ldr.1;
                    b_chan = ldr.2;
                } else {
                    r_chan = ldr.2;
                    g_chan = ldr.1;
                    b_chan = ldr.0;
                }
                a_chan = vdupq_n_u8(255);
            }
            ImageConfiguration::Rgba => {
                let ldr = vld4q_u8(src_ptr);
                r_chan = ldr.0;
                g_chan = ldr.1;
                b_chan = ldr.2;
                a_chan = ldr.3;
            }
            ImageConfiguration::Bgra => {
                let ldr = vld4q_u8(src_ptr);
                r_chan = ldr.2;
                g_chan = ldr.1;
                b_chan = ldr.0;
                a_chan = ldr.3;
            }
        }

        let r_low = vmovl_u8(vget_low_u8(r_chan));
        let g_low = vmovl_u8(vget_low_u8(g_chan));
        let b_low = vmovl_u8(vget_low_u8(b_chan));

        let r_low_low = vmovl_u16(vget_low_u16(r_low));
        let g_low_low = vmovl_u16(vget_low_u16(g_low));
        let b_low_low = vmovl_u16(vget_low_u16(b_low));

        let (x_low_low, y_low_low, z_low_low) =
            neon_triple_to_linear(r_low_low, g_low_low, b_low_low, &transfer);

        let a_low = vmovl_u8(vget_low_u8(a_chan));

        if USE_ALPHA {
            let a_low_low =
                vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_low))), 1f32 / 255f32);
            let xyz_low_low = float32x4x4_t(x_low_low, y_low_low, z_low_low, a_low_low);
            vst4q_f32(dst_ptr.add(cx * channels), xyz_low_low);
        } else {
            let xyz_low_low = float32x4x3_t(x_low_low, y_low_low, z_low_low);
            vst3q_f32(dst_ptr.add(cx * channels), xyz_low_low);
        }

        let r_low_high = vmovl_high_u16(r_low);
        let g_low_high = vmovl_high_u16(g_low);
        let b_low_high = vmovl_high_u16(b_low);

        let (x_low_high, y_low_high, z_low_high) =
            neon_triple_to_linear(r_low_high, g_low_high, b_low_high, &transfer);

        if USE_ALPHA {
            let a_low_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_low)), 1f32 / 255f32);
            let xyz_low_low = float32x4x4_t(x_low_high, y_low_high, z_low_high, a_low_high);
            vst4q_f32(dst_ptr.add(cx * channels + 4 * channels), xyz_low_low);
        } else {
            let xyz_low_low = float32x4x3_t(x_low_high, y_low_high, z_low_high);
            vst3q_f32(dst_ptr.add(cx * channels + 4 * channels), xyz_low_low);
        }

        let r_high = vmovl_high_u8(r_chan);
        let g_high = vmovl_high_u8(g_chan);
        let b_high = vmovl_high_u8(b_chan);

        let r_high_low = vmovl_u16(vget_low_u16(r_high));
        let g_high_low = vmovl_u16(vget_low_u16(g_high));
        let b_high_low = vmovl_u16(vget_low_u16(b_high));

        let (x_high_low, y_high_low, z_high_low) =
            neon_triple_to_linear(r_high_low, g_high_low, b_high_low, &transfer);

        let a_high = vmovl_high_u8(a_chan);

        if USE_ALPHA {
            let a_high_low = vmulq_n_f32(
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_high))),
                1f32 / 255f32,
            );

            let xyz_low_low = float32x4x4_t(x_high_low, y_high_low, z_high_low, a_high_low);
            vst4q_f32(dst_ptr.add(cx * channels + 4 * channels * 2), xyz_low_low);
        } else {
            let xyz_low_low = float32x4x3_t(x_high_low, y_high_low, z_high_low);
            vst3q_f32(dst_ptr.add(cx * channels + 4 * channels * 2), xyz_low_low);
        }

        let r_high_high = vmovl_high_u16(r_high);
        let g_high_high = vmovl_high_u16(g_high);
        let b_high_high = vmovl_high_u16(b_high);

        let (x_high_high, y_high_high, z_high_high) =
            neon_triple_to_linear(r_high_high, g_high_high, b_high_high, &transfer);

        if USE_ALPHA {
            let a_high_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_high)), 1f32 / 255f32);
            let xyz_low_low = float32x4x4_t(x_high_high, y_high_high, z_high_high, a_high_high);
            vst4q_f32(dst_ptr.add(cx * channels + 4 * channels * 3), xyz_low_low);
        } else {
            let xyz_low_low = float32x4x3_t(x_high_high, y_high_high, z_high_high);
            vst3q_f32(dst_ptr.add(cx * channels + 4 * channels * 3), xyz_low_low);
        }

        cx += 16;
    }

    cx
}
