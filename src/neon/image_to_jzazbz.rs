/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::image_to_jzazbz::JzazbzTarget;
use crate::neon::get_neon_linear_transfer;
use crate::neon::math::{vcolorq_matrix_f32, vpowq_n_f32};
use crate::{load_u8_and_deinterleave, TransferFunction, SRGB_TO_XYZ_D65};
use erydanos::{vatan2q_f32, vhypotq_fast_f32, vmlafq_f32};
use std::arch::aarch64::*;

macro_rules! perceptual_quantizer {
    ($color: expr) => {{
        let flush_to_zero_mask = vclezq_f32($color);
        let xx = vpowq_n_f32(vmulq_n_f32($color, 1e-4), 0.1593017578125);
        let jx = vmlafq_f32(vdupq_n_f32(18.8515625), xx, vdupq_n_f32(0.8359375));
        let den_jx = vmlafq_f32(xx, vdupq_n_f32(18.6875), vdupq_n_f32(1.));
        let rs = vpowq_n_f32(vdivq_f32(jx, den_jx), 134.034375);
        vbslq_f32(flush_to_zero_mask, vdupq_n_f32(0.), rs)
    }};
}

macro_rules! triple_to_jzazbz {
    ($r: expr, $g: expr, $b: expr, $transfer: expr, $target: expr, $luminance: expr
    ) => {{
        let r_f = vmulq_n_f32(vcvtq_f32_u32($r), 1f32 / 255f32);
        let g_f = vmulq_n_f32(vcvtq_f32_u32($g), 1f32 / 255f32);
        let b_f = vmulq_n_f32(vcvtq_f32_u32($b), 1f32 / 255f32);
        let dl_l = $transfer(r_f);
        let dl_m = $transfer(g_f);
        let dl_s = $transfer(b_f);

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

        let (mut x, mut y, mut z) = vcolorq_matrix_f32(dl_l, dl_m, dl_s, x0, x1, x2, x3, x4, x5, x6, x7, x8);

        x = vmulq_n_f32(x, $luminance);
        y = vmulq_n_f32(y, $luminance);
        z = vmulq_n_f32(z, $luminance);

        let (l0, l1, l2, l3, l4, l5, l6, l7, l8) = (
            vdupq_n_f32(0.674207838),
            vdupq_n_f32(0.382799340),
            vdupq_n_f32(-0.047570458),
            vdupq_n_f32(0.149284160),
            vdupq_n_f32(0.739628340),
            vdupq_n_f32(0.083327300),
            vdupq_n_f32(0.070941080),
            vdupq_n_f32(0.174768000),
            vdupq_n_f32(0.67097002),
        );

        let (l, m, s) = vcolorq_matrix_f32(x, y, z, l0, l1, l2, l3, l4, l5, l6, l7, l8);

        let lp = perceptual_quantizer!(l);
        let mp = perceptual_quantizer!(m);
        let sp = perceptual_quantizer!(s);

        let iz = vmulq_n_f32(vaddq_f32(lp, mp), 0.5f32);
        let az = vmlafq_f32(
            vdupq_n_f32(3.524000),
            lp,
            vmlafq_f32(vdupq_n_f32(-4.066708), mp, vmulq_n_f32(sp, 0.542708)),
        );
        let bz = vmlafq_f32(
            vdupq_n_f32(0.199076),
            lp,
            vmlafq_f32(vdupq_n_f32(1.096799), mp, vmulq_n_f32(sp, -1.295875)),
        );
        let num = vmulq_n_f32(iz, 0.44);
        let den = vsubq_f32(
            vmlafq_f32(iz, vdupq_n_f32(-0.56), vdupq_n_f32(1.)),
            vdupq_n_f32(1.6295499532821566e-11),
        );
        let jz = vdivq_f32(num, den);

        match $target {
            JzazbzTarget::JZAZBZ => {
                (jz, az, bz)
            }
            JzazbzTarget::JZCZHZ => {
                let cz = vhypotq_fast_f32(az, bz);
                let hz = vatan2q_f32(bz, az);
                (jz, cz, hz)
            }
        }
    }};
}

#[inline(always)]
pub unsafe fn neon_image_to_jzazbz<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> usize {
    let target: JzazbzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer = get_neon_linear_transfer(transfer_function);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_u8_and_deinterleave!(src_ptr, image_configuration);

        let r_low = vmovl_u8(vget_low_u8(r_chan));
        let g_low = vmovl_u8(vget_low_u8(g_chan));
        let b_low = vmovl_u8(vget_low_u8(b_chan));

        let r_low_low = vmovl_u16(vget_low_u16(r_low));
        let g_low_low = vmovl_u16(vget_low_u16(g_low));
        let b_low_low = vmovl_u16(vget_low_u16(b_low));

        let (x_low_low, y_low_low, z_low_low) = triple_to_jzazbz!(
            r_low_low,
            g_low_low,
            b_low_low,
            &transfer,
            target,
            display_luminance
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

        let (x_low_high, y_low_high, z_low_high) = triple_to_jzazbz!(
            r_low_high,
            g_low_high,
            b_low_high,
            &transfer,
            target,
            display_luminance
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

        let (x_high_low, y_high_low, z_high_low) = triple_to_jzazbz!(
            r_high_low,
            g_high_low,
            b_high_low,
            &transfer,
            target,
            display_luminance
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

        let (x_high_high, y_high_high, z_high_high) = triple_to_jzazbz!(
            r_high_high,
            g_high_high,
            b_high_high,
            &transfer,
            target,
            display_luminance
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
