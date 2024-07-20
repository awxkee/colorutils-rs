/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

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
