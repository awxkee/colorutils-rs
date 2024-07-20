/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::neon::get_neon_linear_transfer;
use crate::neon::math::vcolorq_matrix_f32;
use crate::{TransferFunction, SRGB_TO_XYZ_D65};
use erydanos::vcbrtq_fast_f32;
use std::arch::aarch64::*;

macro_rules! triple_to_oklab {
    ($r: expr, $g: expr, $b: expr, $transfer: expr,
    $x0: expr, $x1: expr, $x2: expr, $x3: expr, $x4: expr, $x5: expr, $x6: expr, $x7: expr, $x8: expr,
    $c0:expr, $c1:expr, $c2: expr, $c3: expr, $c4:expr, $c5: expr, $c6:expr, $c7: expr, $c8: expr,
        $m0: expr, $m1: expr, $m2: expr, $m3: expr, $m4: expr, $m5: expr, $m6: expr, $m7: expr, $m8: expr
    ) => {{
        let r_f = vmulq_n_f32(vcvtq_f32_u32($r), 1f32 / 255f32);
        let g_f = vmulq_n_f32(vcvtq_f32_u32($g), 1f32 / 255f32);
        let b_f = vmulq_n_f32(vcvtq_f32_u32($b), 1f32 / 255f32);
        let dl_l = $transfer(r_f);
        let dl_m = $transfer(g_f);
        let dl_s = $transfer(b_f);

        let (x, y, z) = vcolorq_matrix_f32(
            dl_l, dl_m, dl_s, $x0, $x1, $x2, $x3, $x4, $x5, $x6, $x7, $x8,
        );

        let (l_l, l_m, l_s) =
            vcolorq_matrix_f32(x, y, z, $c0, $c1, $c2, $c3, $c4, $c5, $c6, $c7, $c8);

        let l_ = vcbrtq_fast_f32(l_l);
        let m_ = vcbrtq_fast_f32(l_m);
        let s_ = vcbrtq_fast_f32(l_s);

        let (l, m, s) = vcolorq_matrix_f32(l_, m_, s_, $m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, $m8);
        (l, m, s)
    }};
}

#[inline(always)]
pub unsafe fn neon_image_to_oklab<const CHANNELS_CONFIGURATION: u8>(
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

    // Matrix To XYZ
    let (x0, x1, x2, x3, x4, x5, x6, x7, x8) = (
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(0)),
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(1)),
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(2)),
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(0)),
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(1)),
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(2)),
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(0)),
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(1)),
        vdupq_n_f32(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(2)),
    );

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        vdupq_n_f32(0.4122214708f32),
        vdupq_n_f32(0.5363325363f32),
        vdupq_n_f32(0.0514459929f32),
        vdupq_n_f32(0.2119034982f32),
        vdupq_n_f32(0.6806995451f32),
        vdupq_n_f32(0.1073969566f32),
        vdupq_n_f32(0.0883024619f32),
        vdupq_n_f32(0.2817188376f32),
        vdupq_n_f32(0.6299787005f32),
    );

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        vdupq_n_f32(0.2104542553f32),
        vdupq_n_f32(0.7936177850f32),
        vdupq_n_f32(-0.0040720468f32),
        vdupq_n_f32(1.9779984951f32),
        vdupq_n_f32(-2.4285922050f32),
        vdupq_n_f32(0.4505937099f32),
        vdupq_n_f32(0.0259040371f32),
        vdupq_n_f32(0.7827717662f32),
        vdupq_n_f32(-0.8086757660f32),
    );

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

        let (x_low_low, y_low_low, z_low_low) = triple_to_oklab!(
            r_low_low, g_low_low, b_low_low, &transfer, x0, x1, x2, x3, x4, x5, x6, x7, x8, c0, c1,
            c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        let a_low = vmovl_u8(vget_low_u8(a_chan));

        if image_configuration.has_alpha() {
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

        let (x_low_high, y_low_high, z_low_high) = triple_to_oklab!(
            r_low_high, g_low_high, b_low_high, &transfer, x0, x1, x2, x3, x4, x5, x6, x7, x8, c0,
            c1, c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        if image_configuration.has_alpha() {
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

        let (x_high_low, y_high_low, z_high_low) = triple_to_oklab!(
            r_high_low, g_high_low, b_high_low, &transfer, x0, x1, x2, x3, x4, x5, x6, x7, x8, c0,
            c1, c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        let a_high = vmovl_high_u8(a_chan);

        if image_configuration.has_alpha() {
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

        let (x_high_high, y_high_high, z_high_high) = triple_to_oklab!(
            r_high_high,
            g_high_high,
            b_high_high,
            &transfer,
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8
        );

        if image_configuration.has_alpha() {
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
