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
        let row1 = _mm_loadu_si128($ptr as *const __m128i);
        let row2 = _mm_loadu_si128($ptr.add(16) as *const __m128i);
        let row3 = _mm_loadu_si128($ptr.add(32) as *const __m128i);
        match $image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (c1, c2, c3) = sse_deinterleave_rgb(row1, row2, row3);
                if $image_configuration == ImageConfiguration::Rgb {
                    r_chan = c1;
                    g_chan = c2;
                    b_chan = c3;
                } else {
                    r_chan = c3;
                    g_chan = c2;
                    b_chan = c1;
                }
                a_chan = _mm_set1_epi8(-128);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let row4 = _mm_loadu_si128($ptr.add(48) as *const __m128i);
                let (c1, c2, c3, c4) = sse_deinterleave_rgba(row1, row2, row3, row4);
                if $image_configuration == ImageConfiguration::Rgba {
                    r_chan = c1;
                    g_chan = c2;
                    b_chan = c3;
                    a_chan = c4;
                } else {
                    r_chan = c3;
                    g_chan = c2;
                    b_chan = c1;
                    a_chan = c4;
                }
            }
        }
        (r_chan, g_chan, b_chan, a_chan)
    }};
}

#[macro_export]
macro_rules! load_u8_and_deinterleave_half {
    ($ptr: expr, $image_configuration: expr) => {{
        let (r_chan, g_chan, b_chan, a_chan);
        let row1 = _mm_loadu_si128($ptr as *const __m128i);
        let row2 = _mm_loadu_si128($ptr.add(16) as *const __m128i);
        let empty_row = _mm_setzero_si128();
        match $image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (c1, c2, c3) = sse_deinterleave_rgb(row1, row2, empty_row);
                if $image_configuration == ImageConfiguration::Rgb {
                    r_chan = c1;
                    g_chan = c2;
                    b_chan = c3;
                } else {
                    r_chan = c3;
                    g_chan = c2;
                    b_chan = c1;
                }
                a_chan = _mm_set1_epi8(-128);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let (c1, c2, c3, c4) = sse_deinterleave_rgba(row1, row2, empty_row, empty_row);
                if $image_configuration == ImageConfiguration::Rgba {
                    r_chan = c1;
                    g_chan = c2;
                    b_chan = c3;
                    a_chan = c4;
                } else {
                    r_chan = c3;
                    g_chan = c2;
                    b_chan = c1;
                    a_chan = c4;
                }
            }
        }
        (r_chan, g_chan, b_chan, a_chan)
    }};
}

#[macro_export]
macro_rules! load_f32_and_deinterleave {
    ($ptr: expr, $image_configuration: expr) => {{
        let (r_f32, g_f32, b_f32, a_f32);

        let row0 = _mm_loadu_ps($ptr);
        let row1 = _mm_loadu_ps($ptr.add(4));
        let row2 = _mm_loadu_ps($ptr.add(8));

        match $image_configuration {
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let row3 = _mm_loadu_ps($ptr.add(12));
                let (v0, v1, v2, v3) = sse_deinterleave_rgba_ps(row0, row1, row2, row3);
                if $image_configuration == ImageConfiguration::Rgba {
                    r_f32 = v0;
                    g_f32 = v1;
                    b_f32 = v2;
                } else {
                    r_f32 = v2;
                    g_f32 = v1;
                    b_f32 = v0;
                }
                a_f32 = v3;
            }
            ImageConfiguration::Bgr | ImageConfiguration::Rgb => {
                let d_alpha = _mm_set1_ps(1f32);
                let rgb_pixels = sse_deinterleave_rgb_ps(row0, row1, row2);
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
macro_rules! store_and_interleave_v3_direct_f32 {
    ($ptr: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        let (v0, v1, v2) = sse_interleave_ps_rgb($j0, $j1, $j2);
        _mm_storeu_ps($ptr, v0);
        _mm_storeu_ps($ptr.add(4), v1);
        _mm_storeu_ps($ptr.add(8), v2);
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v3_f32 {
    ($ptr: expr, $image_configuration: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        match $image_configuration {
            ImageConfiguration::Rgba => {
                let (v0, v1, v2, v3) = sse_interleave_ps_rgba($j0, $j1, $j2, _mm_setzero_ps());
                _mm_storeu_ps($ptr, v0);
                _mm_storeu_ps($ptr.add(4), v1);
                _mm_storeu_ps($ptr.add(8), v2);
                _mm_storeu_ps($ptr.add(12), v3);
            }
            ImageConfiguration::Bgra => {
                let (v0, v1, v2, v3) = sse_interleave_ps_rgba($j2, $j1, $j0, _mm_setzero_ps());
                _mm_storeu_ps($ptr, v0);
                _mm_storeu_ps($ptr.add(4), v1);
                _mm_storeu_ps($ptr.add(8), v2);
                _mm_storeu_ps($ptr.add(12), v3);
            }
            ImageConfiguration::Rgb => {
                let (v0, v1, v2) = sse_interleave_ps_rgb($j0, $j1, $j2);
                _mm_storeu_ps($ptr, v0);
                _mm_storeu_ps($ptr.add(4), v1);
                _mm_storeu_ps($ptr.add(8), v2);
            }
            ImageConfiguration::Bgr => {
                let (v0, v1, v2) = sse_interleave_ps_rgb($j2, $j1, $j0);
                _mm_storeu_ps($ptr, v0);
                _mm_storeu_ps($ptr.add(4), v1);
                _mm_storeu_ps($ptr.add(8), v2);
            }
        }
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v4_direct_f32 {
    ($ptr: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        let (v0, v1, v2, v3) = sse_interleave_ps_rgba($j0, $j1, $j2, $j3);
        _mm_storeu_ps($ptr, v0);
        _mm_storeu_ps($ptr.add(4), v1);
        _mm_storeu_ps($ptr.add(8), v2);
        _mm_storeu_ps($ptr.add(12), v3);
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v4_f32 {
    ($ptr: expr, $image_configuration: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        match $image_configuration {
            ImageConfiguration::Rgba => {
                let (v0, v1, v2, v3) = sse_interleave_ps_rgba($j0, $j1, $j2, $j3);
                _mm_storeu_ps($ptr, v0);
                _mm_storeu_ps($ptr.add(4), v1);
                _mm_storeu_ps($ptr.add(8), v2);
                _mm_storeu_ps($ptr.add(12), v3);
            }
            ImageConfiguration::Bgra => {
                let (v0, v1, v2, v3) = sse_interleave_ps_rgba($j2, $j1, $j0, $j3);
                _mm_storeu_ps($ptr, v0);
                _mm_storeu_ps($ptr.add(4), v1);
                _mm_storeu_ps($ptr.add(8), v2);
                _mm_storeu_ps($ptr.add(12), v3);
            }
            ImageConfiguration::Rgb => {
                let (v0, v1, v2) = sse_interleave_ps_rgb($j0, $j1, $j2);
                _mm_storeu_ps($ptr, v0);
                _mm_storeu_ps($ptr.add(4), v1);
                _mm_storeu_ps($ptr.add(8), v2);
            }
            ImageConfiguration::Bgr => {
                let (v0, v1, v2) = sse_interleave_ps_rgb($j2, $j1, $j0);
                _mm_storeu_ps($ptr, v0);
                _mm_storeu_ps($ptr.add(4), v1);
                _mm_storeu_ps($ptr.add(8), v2);
            }
        }
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v4_u8 {
    ($ptr: expr, $configuration: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        match $configuration {
            ImageConfiguration::Rgba => {
                let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba($j0, $j1, $j2, $j3);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
                _mm_storeu_si128($ptr.add(48) as *mut __m128i, rgba3);
            }
            ImageConfiguration::Bgra => {
                let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba($j2, $j1, $j0, $j3);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
                _mm_storeu_si128($ptr.add(48) as *mut __m128i, rgba3);
            }
            ImageConfiguration::Rgb => {
                let (rgba0, rgba1, rgba2) = sse_interleave_rgb($j0, $j1, $j2);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
            }
            ImageConfiguration::Bgr => {
                let (rgba0, rgba1, rgba2) = sse_interleave_rgb($j2, $j1, $j0);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
            }
        }
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v4_half_u8 {
    ($ptr: expr, $configuration: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        match $configuration {
            ImageConfiguration::Rgba => {
                let (rgba0, rgba1, _, _) = sse_interleave_rgba($j0, $j1, $j2, $j3);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
            }
            ImageConfiguration::Bgra => {
                let (rgba0, rgba1, _, _) = sse_interleave_rgba($j2, $j1, $j0, $j3);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
            }
            ImageConfiguration::Rgb => {
                let (rgba0, rgba1, _) = sse_interleave_rgb($j0, $j1, $j2);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                std::ptr::copy_nonoverlapping(&rgba1 as *const _ as *const u8, $ptr.add(16), 8);
            }
            ImageConfiguration::Bgr => {
                let (rgba0, rgba1, _) = sse_interleave_rgb($j2, $j1, $j0);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                std::ptr::copy_nonoverlapping(&rgba1 as *const _ as *const u8, $ptr.add(16), 8);
            }
        }
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v4_direct_u8 {
    ($ptr: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba($j0, $j1, $j2, $j3);
        _mm_storeu_si128($ptr as *mut __m128i, rgba0);
        _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
        _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
        _mm_storeu_si128($ptr.add(48) as *mut __m128i, rgba3);
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v3_u8 {
    ($ptr: expr, $image_configuration: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        match $image_configuration {
            ImageConfiguration::Rgba => {
                let (rgba0, rgba1, rgba2, rgba3) =
                    sse_interleave_rgba($j0, $j1, $j2, _mm_setzero_si128());
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
                _mm_storeu_si128($ptr.add(48) as *mut __m128i, rgba3);
            }
            ImageConfiguration::Bgra => {
                let (rgba0, rgba1, rgba2, rgba3) =
                    sse_interleave_rgba($j2, $j1, $j0, _mm_setzero_si128());
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
                _mm_storeu_si128($ptr.add(48) as *mut __m128i, rgba3);
            }
            ImageConfiguration::Rgb => {
                let (rgba0, rgba1, rgba2) = sse_interleave_rgb($j0, $j1, $j2);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
            }
            ImageConfiguration::Bgr => {
                let (rgba0, rgba1, rgba2) = sse_interleave_rgb($j2, $j1, $j0);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
            }
        }
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v3_half_u8 {
    ($ptr: expr, $image_configuration: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        match $image_configuration {
            ImageConfiguration::Rgba => {
                let (rgba0, rgba1, _, _) = sse_interleave_rgba($j0, $j1, $j2, _mm_setzero_si128());
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
            }
            ImageConfiguration::Bgra => {
                let (rgba0, rgba1, _, _) = sse_interleave_rgba($j2, $j1, $j0, _mm_setzero_si128());
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
            }
            ImageConfiguration::Rgb => {
                let (rgba0, rgba1, _) = sse_interleave_rgb($j0, $j1, $j2);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                std::ptr::copy_nonoverlapping(&rgba1 as *const _ as *const u8, $ptr.add(16), 8);
            }
            ImageConfiguration::Bgr => {
                let (rgba0, rgba1, _) = sse_interleave_rgb($j2, $j1, $j0);
                _mm_storeu_si128($ptr as *mut __m128i, rgba0);
                std::ptr::copy_nonoverlapping(&rgba1 as *const _ as *const u8, $ptr.add(16), 8);
            }
        }
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v3_direct_u8 {
    ($ptr: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        let (rgba0, rgba1, rgba2) = sse_interleave_rgb($j0, $j1, $j2);
        _mm_storeu_si128($ptr as *mut __m128i, rgba0);
        _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba1);
        _mm_storeu_si128($ptr.add(32) as *mut __m128i, rgba2);
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v4_u16 {
    ($ptr: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba_epi16($j0, $j1, $j2, $j3);
        _mm_storeu_si128($ptr as *mut __m128i, rgba0);
        _mm_storeu_si128($ptr.add(8) as *mut __m128i, rgba1);
        _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba2);
        _mm_storeu_si128($ptr.add(224) as *mut __m128i, rgba3);
    }};
}

#[macro_export]
macro_rules! store_and_interleave_v3_u16 {
    ($ptr: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        let (rgba0, rgba1, rgba2) = sse_interleave_rgb_epi16($j0, $j1, $j2);
        _mm_storeu_si128($ptr as *mut __m128i, rgba0);
        _mm_storeu_si128($ptr.add(8) as *mut __m128i, rgba1);
        _mm_storeu_si128($ptr.add(16) as *mut __m128i, rgba2);
    }};
}
