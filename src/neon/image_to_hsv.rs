/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
use crate::neon::{neon_rgb_to_hsl, neon_rgb_to_hsv};
use crate::{load_u8_and_deinterleave, load_u8_and_deinterleave_half};
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn neon_channels_to_hsv_u16<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: &[u8],
    width: u32,
    dst: &mut [u16],
    scale: f32,
) -> usize {
    let target: HsvTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let mut cx = start_cx;
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Use alpha flag used on image without alpha");
    }

    let channels = image_configuration.channel_count();

    let v_scale = vdupq_n_f32(scale);

    while cx + 16 <= width as usize {
        unsafe {
            let src_ptr = src.get_unchecked(cx * channels);
            let (r_chan, g_chan, b_chan, a_chan) =
                load_u8_and_deinterleave!(src_ptr, image_configuration);

            let r_low = vmovl_u8(vget_low_u8(r_chan));
            let g_low = vmovl_u8(vget_low_u8(g_chan));
            let b_low = vmovl_u8(vget_low_u8(b_chan));

            let r_low_low = vmovl_u16(vget_low_u16(r_low));
            let g_low_low = vmovl_u16(vget_low_u16(g_low));
            let b_low_low = vmovl_u16(vget_low_u16(b_low));

            let (x_low_low, y_low_low, z_low_low) = match target {
                HsvTarget::Hsv => neon_rgb_to_hsv(r_low_low, g_low_low, b_low_low, v_scale),
                HsvTarget::Hsl => neon_rgb_to_hsl(r_low_low, g_low_low, b_low_low, v_scale),
            };

            let a_low = vmovl_u8(vget_low_u8(a_chan));

            let r_low_high = vmovl_high_u16(r_low);
            let g_low_high = vmovl_high_u16(g_low);
            let b_low_high = vmovl_high_u16(b_low);

            let (x_low_high, y_low_high, z_low_high) = match target {
                HsvTarget::Hsv => neon_rgb_to_hsv(r_low_high, g_low_high, b_low_high, v_scale),
                HsvTarget::Hsl => neon_rgb_to_hsl(r_low_high, g_low_high, b_low_high, v_scale),
            };

            let x_low = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(x_low_low)),
                vmovn_u32(vcvtaq_u32_f32(x_low_high)),
            );
            let y_low = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(y_low_low)),
                vmovn_u32(vcvtaq_u32_f32(y_low_high)),
            );
            let z_low = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(z_low_low)),
                vmovn_u32(vcvtaq_u32_f32(z_low_high)),
            );

            if USE_ALPHA {
                let xyz_low_low = uint16x8x4_t(x_low, y_low, z_low, a_low);
                vst4q_u16(
                    dst.get_unchecked_mut(cx * channels..).as_mut_ptr(),
                    xyz_low_low,
                );
            } else {
                let xyz_low_low = uint16x8x3_t(x_low, y_low, z_low);
                vst3q_u16(
                    dst.get_unchecked_mut(cx * channels..).as_mut_ptr(),
                    xyz_low_low,
                );
            }

            let r_high = vmovl_high_u8(r_chan);
            let g_high = vmovl_high_u8(g_chan);
            let b_high = vmovl_high_u8(b_chan);

            let r_high_low = vmovl_u16(vget_low_u16(r_high));
            let g_high_low = vmovl_u16(vget_low_u16(g_high));
            let b_high_low = vmovl_u16(vget_low_u16(b_high));

            let (x_high_low, y_high_low, z_high_low) = match target {
                HsvTarget::Hsv => neon_rgb_to_hsv(r_high_low, g_high_low, b_high_low, v_scale),
                HsvTarget::Hsl => neon_rgb_to_hsl(r_high_low, g_high_low, b_high_low, v_scale),
            };

            let a_high = vmovl_high_u8(a_chan);

            let r_high_high = vmovl_high_u16(r_high);
            let g_high_high = vmovl_high_u16(g_high);
            let b_high_high = vmovl_high_u16(b_high);

            let (x_high_high, y_high_high, z_high_high) = match target {
                HsvTarget::Hsv => neon_rgb_to_hsv(r_high_high, g_high_high, b_high_high, v_scale),
                HsvTarget::Hsl => neon_rgb_to_hsl(r_high_high, g_high_high, b_high_high, v_scale),
            };

            let x_high = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(x_high_low)),
                vmovn_u32(vcvtaq_u32_f32(x_high_high)),
            );
            let y_high = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(y_high_low)),
                vmovn_u32(vcvtaq_u32_f32(y_high_high)),
            );
            let z_high = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(z_high_low)),
                vmovn_u32(vcvtaq_u32_f32(z_high_high)),
            );

            if USE_ALPHA {
                let xyz_low_low = uint16x8x4_t(x_high, y_high, z_high, a_high);
                vst4q_u16(
                    dst.get_unchecked_mut(cx * channels + 8 * channels..)
                        .as_mut_ptr(),
                    xyz_low_low,
                );
            } else {
                let xyz_low_low = uint16x8x3_t(x_high, y_high, z_high);
                vst3q_u16(
                    dst.get_unchecked_mut(cx * channels + 8 * channels..)
                        .as_mut_ptr(),
                    xyz_low_low,
                );
            }
        }

        cx += 16;
    }

    while cx + 8 <= width as usize {
        unsafe {
            let src_ptr = src.get_unchecked(cx * channels..).as_ptr();

            let (r_chan, g_chan, b_chan, a_chan) =
                load_u8_and_deinterleave_half!(src_ptr, image_configuration);

            let r_low = vmovl_u8(vget_low_u8(r_chan));
            let g_low = vmovl_u8(vget_low_u8(g_chan));
            let b_low = vmovl_u8(vget_low_u8(b_chan));

            let r_low_low = vmovl_u16(vget_low_u16(r_low));
            let g_low_low = vmovl_u16(vget_low_u16(g_low));
            let b_low_low = vmovl_u16(vget_low_u16(b_low));

            let (x_low_low, y_low_low, z_low_low) = match target {
                HsvTarget::Hsv => neon_rgb_to_hsv(r_low_low, g_low_low, b_low_low, v_scale),
                HsvTarget::Hsl => neon_rgb_to_hsl(r_low_low, g_low_low, b_low_low, v_scale),
            };

            let a_low = vmovl_u8(vget_low_u8(a_chan));

            let r_low_high = vmovl_high_u16(r_low);
            let g_low_high = vmovl_high_u16(g_low);
            let b_low_high = vmovl_high_u16(b_low);

            let (x_low_high, y_low_high, z_low_high) = match target {
                HsvTarget::Hsv => neon_rgb_to_hsv(r_low_high, g_low_high, b_low_high, v_scale),
                HsvTarget::Hsl => neon_rgb_to_hsl(r_low_high, g_low_high, b_low_high, v_scale),
            };

            let x_low = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(x_low_low)),
                vmovn_u32(vcvtaq_u32_f32(x_low_high)),
            );
            let y_low = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(y_low_low)),
                vmovn_u32(vcvtaq_u32_f32(y_low_high)),
            );
            let z_low = vcombine_u16(
                vmovn_u32(vcvtaq_u32_f32(z_low_low)),
                vmovn_u32(vcvtaq_u32_f32(z_low_high)),
            );

            if USE_ALPHA {
                let xyz_low_low = uint16x8x4_t(x_low, y_low, z_low, a_low);
                vst4q_u16(
                    dst.get_unchecked_mut(cx * channels..).as_mut_ptr(),
                    xyz_low_low,
                );
            } else {
                let xyz_low_low = uint16x8x3_t(x_low, y_low, z_low);
                vst3q_u16(
                    dst.get_unchecked_mut(cx * channels..).as_mut_ptr(),
                    xyz_low_low,
                );
            }
        }

        cx += 8;
    }

    cx
}
