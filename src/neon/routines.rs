/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[macro_export]
macro_rules! load_u8_and_deinterleave {
    ($ptr: expr, $image_configuration: expr) => {{
        let (r_chan, g_chan, b_chan, a_chan);
        match $image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let ldr = vld3q_u8($ptr);
                if $image_configuration == ImageConfiguration::Rgb {
                    r_chan = ldr.0;
                    g_chan = ldr.1;
                    b_chan = ldr.2;
                } else {
                    r_chan = ldr.2;
                    g_chan = ldr.1;
                    b_chan = ldr.0;
                }
                a_chan = vdupq_n_u8(0);
            }
            ImageConfiguration::Rgba => {
                let ldr = vld4q_u8($ptr);
                r_chan = ldr.0;
                g_chan = ldr.1;
                b_chan = ldr.2;
                a_chan = ldr.3;
            }
            ImageConfiguration::Bgra => {
                let ldr = vld4q_u8($ptr);
                r_chan = ldr.2;
                g_chan = ldr.1;
                b_chan = ldr.0;
                a_chan = ldr.3;
            }
        }
        (r_chan, g_chan, b_chan, a_chan)
    }};
}

#[macro_export]
macro_rules! load_u8_and_deinterleave_half {
    ($ptr: expr, $image_configuration: expr) => {{
        let (r_chan, g_chan, b_chan, a_chan);
        match $image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let ldr = vld3_u8($ptr);
                if $image_configuration == ImageConfiguration::Rgb {
                    r_chan = ldr.0;
                    g_chan = ldr.1;
                    b_chan = ldr.2;
                } else {
                    r_chan = ldr.2;
                    g_chan = ldr.1;
                    b_chan = ldr.0;
                }
                a_chan = vdup_n_u8(0);
            }
            ImageConfiguration::Rgba => {
                let ldr = vld4_u8($ptr);
                r_chan = ldr.0;
                g_chan = ldr.1;
                b_chan = ldr.2;
                a_chan = ldr.3;
            }
            ImageConfiguration::Bgra => {
                let ldr = vld4_u8($ptr);
                r_chan = ldr.2;
                g_chan = ldr.1;
                b_chan = ldr.0;
                a_chan = ldr.3;
            }
        }
        let zero_lane = vdup_n_u8(0);
        (
            vcombine_u8(r_chan, zero_lane),
            vcombine_u8(g_chan, zero_lane),
            vcombine_u8(b_chan, zero_lane),
            vcombine_u8(a_chan, zero_lane),
        )
    }};
}

#[macro_export]
macro_rules! load_f32_and_deinterleave {
    ($ptr: expr, $image_configuration: expr) => {{
        let d_alpha = vdupq_n_f32(1f32);
        let (r_f32, g_f32, b_f32, a_f32);
        match $image_configuration {
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let rgba_pixels = vld4q_f32($ptr);
                if $image_configuration == ImageConfiguration::Rgba {
                    r_f32 = rgba_pixels.0;
                    g_f32 = rgba_pixels.1;
                    b_f32 = rgba_pixels.2;
                } else {
                    r_f32 = rgba_pixels.2;
                    g_f32 = rgba_pixels.1;
                    b_f32 = rgba_pixels.0;
                }
                a_f32 = rgba_pixels.3;
            }
            ImageConfiguration::Bgr | ImageConfiguration::Rgb => {
                let rgb_pixels = vld3q_f32($ptr);
                if $image_configuration == ImageConfiguration::Rgb {
                    r_f32 = rgb_pixels.0;
                    g_f32 = rgb_pixels.1;
                    b_f32 = rgb_pixels.2;
                } else {
                    r_f32 = rgb_pixels.2;
                    g_f32 = rgb_pixels.1;
                    b_f32 = rgb_pixels.0;
                }
                a_f32 = d_alpha;
            }
        }
        (r_f32, g_f32, b_f32, a_f32)
    }};
}

#[macro_export]
macro_rules! load_f32_and_deinterleave_direct {
    ($ptr: expr, $image_configuration: expr) => {{
        let d_alpha = vdupq_n_f32(1f32);
        let (r_f32, g_f32, b_f32, a_f32);
        match $image_configuration {
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let rgba_pixels = vld4q_f32($ptr);
                r_f32 = rgba_pixels.0;
                g_f32 = rgba_pixels.1;
                b_f32 = rgba_pixels.2;
                a_f32 = rgba_pixels.3;
            }
            ImageConfiguration::Bgr | ImageConfiguration::Rgb => {
                let rgb_pixels = vld3q_f32($ptr);
                r_f32 = rgb_pixels.0;
                g_f32 = rgb_pixels.1;
                b_f32 = rgb_pixels.2;
                a_f32 = d_alpha;
            }
        }
        (r_f32, g_f32, b_f32, a_f32)
    }};
}
