/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
use crate::neon::{neon_hsl_to_rgb, neon_hsv_to_rgb};

#[inline]
pub unsafe fn neon_hsv_u16_to_image<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: *const u16,
    src_offset: usize,
    width: u32,
    dst: *mut u8,
    dst_offset: usize,
    scale: f32,
) -> usize {
    let target: HsvTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let mut cx = start_cx;
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Use alpha flag used on image without alpha");
        }
    }

    let channels = image_configuration.get_channels_count();
    let v_scale = vdupq_n_f32(scale);
    let dst_ptr = dst.add(dst_offset);

    while cx + 16 < width as usize {
        let (h_chan, s_chan, v_chan, a_chan_lo);
        let src_ptr = ((src as *const u8).add(src_offset) as *const u16).add(cx * channels);

        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let hsv_pixel = vld3q_u16(src_ptr);
                h_chan = hsv_pixel.0;
                s_chan = hsv_pixel.1;
                v_chan = hsv_pixel.2;
                a_chan_lo = vdupq_n_u16(255);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let hsv_pixel = vld4q_u16(src_ptr);
                h_chan = hsv_pixel.0;
                s_chan = hsv_pixel.1;
                v_chan = hsv_pixel.2;
                a_chan_lo = hsv_pixel.3;
            }
        }

        let h_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(h_chan)));
        let s_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(s_chan)));
        let v_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_chan)));

        let (r_low, g_low, b_low) = match target {
            HsvTarget::HSV => neon_hsv_to_rgb(h_low, s_low, v_low, v_scale),
            HsvTarget::HSL => neon_hsl_to_rgb(h_low, s_low, v_low, v_scale),
        };

        let h_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(h_chan)));
        let s_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(s_chan)));
        let v_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_chan)));

        let (r_high, g_high, b_high) = match target {
            HsvTarget::HSV => neon_hsv_to_rgb(h_high, s_high, v_high, v_scale),
            HsvTarget::HSL => neon_hsl_to_rgb(h_high, s_high, v_high, v_scale),
        };

        let r_chan_16_lo = vcombine_u16(vmovn_u32(r_low), vmovn_u32(r_high));
        let g_chan_16_lo = vcombine_u16(vmovn_u32(g_low), vmovn_u32(g_high));
        let b_chan_16_lo = vcombine_u16(vmovn_u32(b_low), vmovn_u32(b_high));

        let src_ptr = src_ptr.add(8 * channels);
        let (h_chan, s_chan, v_chan, a_chan_hi);
        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let hsv_pixel = vld3q_u16(src_ptr);
                h_chan = hsv_pixel.0;
                s_chan = hsv_pixel.1;
                v_chan = hsv_pixel.2;
                a_chan_hi = vdupq_n_u16(255);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let hsv_pixel = vld4q_u16(src_ptr);
                h_chan = hsv_pixel.0;
                s_chan = hsv_pixel.1;
                v_chan = hsv_pixel.2;
                a_chan_hi = hsv_pixel.3;
            }
        }

        let h_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(h_chan)));
        let s_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(s_chan)));
        let v_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_chan)));

        let (r_low, g_low, b_low) = match target {
            HsvTarget::HSV => neon_hsv_to_rgb(h_low, s_low, v_low, v_scale),
            HsvTarget::HSL => neon_hsl_to_rgb(h_low, s_low, v_low, v_scale),
        };

        let h_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(h_chan)));
        let s_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(s_chan)));
        let v_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_chan)));

        let (r_high, g_high, b_high) = match target {
            HsvTarget::HSV => neon_hsv_to_rgb(h_high, s_high, v_high, v_scale),
            HsvTarget::HSL => neon_hsl_to_rgb(h_high, s_high, v_high, v_scale),
        };

        let r_chan_16_hi = vcombine_u16(vmovn_u32(r_low), vmovn_u32(r_high));
        let g_chan_16_hi = vcombine_u16(vmovn_u32(g_low), vmovn_u32(g_high));
        let b_chan_16_hi = vcombine_u16(vmovn_u32(b_low), vmovn_u32(b_high));

        let r_chan = vcombine_u8(vqmovn_u16(r_chan_16_lo), vqmovn_u16(r_chan_16_hi));
        let g_chan = vcombine_u8(vqmovn_u16(g_chan_16_lo), vqmovn_u16(g_chan_16_hi));
        let b_chan = vcombine_u8(vqmovn_u16(b_chan_16_lo), vqmovn_u16(b_chan_16_hi));

        if USE_ALPHA {
            let a_chan = vcombine_u8(vqmovn_u16(a_chan_lo), vqmovn_u16(a_chan_hi));
            let pixel_set = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x16x4_t(r_chan, g_chan, b_chan, a_chan)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x16x4_t(b_chan, g_chan, r_chan, a_chan)
                }
            };
            vst4q_u8(dst_ptr.add(cx * channels), pixel_set);
        } else {
            let pixel_set = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x16x3_t(r_chan, g_chan, b_chan)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x16x3_t(b_chan, g_chan, r_chan)
                }
            };
            vst3q_u8(dst_ptr.add(cx * channels), pixel_set);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let (h_chan, s_chan, v_chan, a_chan);
        let src_ptr = ((src as *const u8).add(src_offset) as *const u16).add(cx * channels);

        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let hsv_pixel = vld3q_u16(src_ptr);
                h_chan = hsv_pixel.0;
                s_chan = hsv_pixel.1;
                v_chan = hsv_pixel.2;
                a_chan = vdupq_n_u16(255);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let hsv_pixel = vld4q_u16(src_ptr);
                h_chan = hsv_pixel.0;
                s_chan = hsv_pixel.1;
                v_chan = hsv_pixel.2;
                a_chan = hsv_pixel.3;
            }
        }

        let h_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(h_chan)));
        let s_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(s_chan)));
        let v_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_chan)));

        let (r_low, g_low, b_low) = match target {
            HsvTarget::HSV => neon_hsv_to_rgb(h_low, s_low, v_low, v_scale),
            HsvTarget::HSL => neon_hsl_to_rgb(h_low, s_low, v_low, v_scale),
        };

        let h_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(h_chan)));
        let s_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(s_chan)));
        let v_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_chan)));

        let (r_high, g_high, b_high) = match target {
            HsvTarget::HSV => neon_hsv_to_rgb(h_high, s_high, v_high, v_scale),
            HsvTarget::HSL => neon_hsl_to_rgb(h_high, s_high, v_high, v_scale),
        };

        let r_chan_16 = vcombine_u16(vmovn_u32(r_low), vmovn_u32(r_high));
        let g_chan_16 = vcombine_u16(vmovn_u32(g_low), vmovn_u32(g_high));
        let b_chan_16 = vcombine_u16(vmovn_u32(b_low), vmovn_u32(b_high));
        let r_chan = vqmovn_u16(r_chan_16);
        let g_chan = vqmovn_u16(g_chan_16);
        let b_chan = vqmovn_u16(b_chan_16);

        if USE_ALPHA {
            let a_chan = vqmovn_u16(a_chan);
            match image_configuration {
                ImageConfiguration::Rgb => {}
                ImageConfiguration::Rgba => {
                    let pixel_set = uint8x8x4_t(r_chan, g_chan, b_chan, a_chan);
                    vst4_u8(dst_ptr.add(cx * channels), pixel_set);
                }
                ImageConfiguration::Bgra => {
                    let pixel_set = uint8x8x4_t(b_chan, g_chan, r_chan, a_chan);
                    vst4_u8(dst_ptr.add(cx * channels), pixel_set);
                }
                ImageConfiguration::Bgr => {}
            }
        } else {
            match image_configuration {
                ImageConfiguration::Rgb => {
                    let pixel_set = uint8x8x3_t(r_chan, g_chan, b_chan);
                    vst3_u8(dst_ptr.add(cx * channels), pixel_set);
                }
                ImageConfiguration::Rgba => {}
                ImageConfiguration::Bgra => {}
                ImageConfiguration::Bgr => {
                    let pixel_set = uint8x8x3_t(b_chan, g_chan, r_chan);
                    vst3_u8(dst_ptr.add(cx * channels), pixel_set);
                }
            }
        }

        cx += 8;
    }

    cx
}
