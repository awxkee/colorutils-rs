/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::avx::{
    avx2_deinterleave_rgb_epi8, avx2_deinterleave_rgb_ps, avx2_deinterleave_rgba_epi8,
    avx2_deinterleave_rgba_ps,
};
use crate::image::ImageConfiguration;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub(crate) unsafe fn avx_vld_u8_and_deinterleave<const CHANNELS_CONFIGURATION: u8>(
    ptr: *const u8,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let (r_chan, g_chan, b_chan, a_chan);

    let row1 = _mm256_loadu_si256(ptr as *const __m256i);
    let row2 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
    let row3 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
    match image_configuration {
        ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
            let (c1, c2, c3) = avx2_deinterleave_rgb_epi8(row1, row2, row3);
            if image_configuration == ImageConfiguration::Rgb {
                r_chan = c1;
                g_chan = c2;
                b_chan = c3;
            } else {
                r_chan = c3;
                g_chan = c2;
                b_chan = c1;
            }
            a_chan = _mm256_set1_epi8(-128);
        }
        ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
            let row4 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);
            let (c1, c2, c3, c4) = avx2_deinterleave_rgba_epi8(row1, row2, row3, row4);
            if image_configuration == ImageConfiguration::Rgba {
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
}

#[inline]
pub(crate) unsafe fn avx_vld_u8_and_deinterleave_half<const CHANNELS_CONFIGURATION: u8>(
    ptr: *const u8,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let (r_chan, g_chan, b_chan, a_chan);

    let row1 = _mm256_loadu_si256(ptr as *const __m256i);
    let row2 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
    let empty_row = _mm256_setzero_si256();
    match image_configuration {
        ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
            let (c1, c2, c3) = avx2_deinterleave_rgb_epi8(row1, row2, empty_row);
            if image_configuration == ImageConfiguration::Rgb {
                r_chan = c1;
                g_chan = c2;
                b_chan = c3;
            } else {
                r_chan = c3;
                g_chan = c2;
                b_chan = c1;
            }
            a_chan = _mm256_set1_epi8(-128);
        }
        ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
            let (c1, c2, c3, c4) = avx2_deinterleave_rgba_epi8(row1, row2, empty_row, empty_row);
            if image_configuration == ImageConfiguration::Rgba {
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
}

#[inline]
pub(crate) unsafe fn avx_vld_u8_and_deinterleave_quarter<const CHANNELS_CONFIGURATION: u8>(
    ptr: *const u8,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let (r_chan, g_chan, b_chan, a_chan);

    let row1 = _mm256_loadu_si256(ptr as *const __m256i);
    let empty_row = _mm256_setzero_si256();
    match image_configuration {
        ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
            let (c1, c2, c3) = avx2_deinterleave_rgb_epi8(row1, empty_row, empty_row);
            if image_configuration == ImageConfiguration::Rgb {
                r_chan = c1;
                g_chan = c2;
                b_chan = c3;
            } else {
                r_chan = c3;
                g_chan = c2;
                b_chan = c1;
            }
            a_chan = _mm256_set1_epi8(-128);
        }
        ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
            let (c1, c2, c3, c4) =
                avx2_deinterleave_rgba_epi8(row1, empty_row, empty_row, empty_row);
            if image_configuration == ImageConfiguration::Rgba {
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
}

#[inline]
pub(crate) unsafe fn avx_vld_f32_and_deinterleave<const CHANNELS_CONFIGURATION: u8>(
    ptr: *const f32,
) -> (__m256, __m256, __m256, __m256) {
    let (r_f32, g_f32, b_f32, a_f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let row0 = _mm256_loadu_ps(ptr);
    let row1 = _mm256_loadu_ps(ptr.add(8));
    let row2 = _mm256_loadu_ps(ptr.add(16));

    match image_configuration {
        ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
            let row3 = _mm256_loadu_ps(ptr.add(24));
            let (v0, v1, v2, v3) = avx2_deinterleave_rgba_ps(row0, row1, row2, row3);
            if image_configuration == ImageConfiguration::Rgba {
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
            let rgb_pixels = avx2_deinterleave_rgb_ps(row0, row1, row2);
            if image_configuration == ImageConfiguration::Rgb {
                r_f32 = rgb_pixels.0;
                g_f32 = rgb_pixels.1;
                b_f32 = rgb_pixels.2;
            } else {
                r_f32 = rgb_pixels.2;
                g_f32 = rgb_pixels.1;
                b_f32 = rgb_pixels.0;
            }
            a_f32 = _mm256_set1_ps(1.);
        }
    }

    (r_f32, g_f32, b_f32, a_f32)
}

#[macro_export]
macro_rules! avx_store_and_interleave_u8 {
    ($ptr: expr, $configuration: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        match $configuration {
            ImageConfiguration::Rgb => {
                let (rgb0, rgb1, rgb2) = avx2_interleave_rgb($j0, $j1, $j2);
                _mm256_storeu_si256($ptr as *mut __m256i, rgb0);
                _mm256_storeu_si256($ptr.add(32) as *mut __m256i, rgb1);
                _mm256_storeu_si256($ptr.add(64) as *mut __m256i, rgb2);
            }
            ImageConfiguration::Rgba => {
                let (rgba0, rgba1, rgba2, rgba3) = avx2_interleave_rgba_epi8($j0, $j1, $j2, $j3);
                _mm256_storeu_si256($ptr as *mut __m256i, rgba0);
                _mm256_storeu_si256($ptr.add(32) as *mut __m256i, rgba1);
                _mm256_storeu_si256($ptr.add(64) as *mut __m256i, rgba2);
                _mm256_storeu_si256($ptr.add(96) as *mut __m256i, rgba3);
            }
            ImageConfiguration::Bgra => {
                let (bgra0, bgra1, bgra2, bgra3) = avx2_interleave_rgba_epi8($j2, $j1, $j0, $j3);
                _mm256_storeu_si256($ptr as *mut __m256i, bgra0);
                _mm256_storeu_si256($ptr.add(32) as *mut __m256i, bgra1);
                _mm256_storeu_si256($ptr.add(64) as *mut __m256i, bgra2);
                _mm256_storeu_si256($ptr.add(96) as *mut __m256i, bgra3);
            }
            ImageConfiguration::Bgr => {
                let (bgr0, bgr1, bgr2) = avx2_interleave_rgb($j2, $j1, $j0);
                _mm256_storeu_si256($ptr as *mut __m256i, bgr0);
                _mm256_storeu_si256($ptr.add(32) as *mut __m256i, bgr1);
                _mm256_storeu_si256($ptr.add(64) as *mut __m256i, bgr2);
            }
        }
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v4_u8 {
    ($ptr: expr, $configuration: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        let (row0, row1, row2, row3);
        match $configuration {
            ImageConfiguration::Rgba => {
                (row0, row1, row2, row3) = avx2_interleave_rgba_epi8($j0, $j1, $j2, $j3)
            }
            ImageConfiguration::Rgb => {
                (row0, row1, row2) = avx2_interleave_rgb($j0, $j1, $j2);
                row3 = _mm256_setzero_si256();
            }
            ImageConfiguration::Bgr => {
                (row0, row1, row2) = avx2_interleave_rgb($j2, $j1, $j0);
                row3 = _mm256_setzero_si256();
            }
            ImageConfiguration::Bgra => {
                (row0, row1, row2, row3) = avx2_interleave_rgba_epi8($j2, $j1, $j0, $j3)
            }
        };
        _mm256_storeu_si256($ptr as *mut __m256i, row0);
        _mm256_storeu_si256($ptr.add(32) as *mut __m256i, row1);
        _mm256_storeu_si256($ptr.add(64) as *mut __m256i, row2);
        _mm256_storeu_si256($ptr.add(96) as *mut __m256i, row3);
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v4_half_u8 {
    ($ptr: expr, $configuration: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        let (row0, row1);
        match $configuration {
            ImageConfiguration::Rgba => {
                (row0, row1, _, _) = avx2_interleave_rgba_epi8($j0, $j1, $j2, $j3)
            }
            ImageConfiguration::Rgb => {
                (row0, row1, _) = avx2_interleave_rgb($j0, $j1, $j2);
            }
            ImageConfiguration::Bgr => {
                (row0, row1, _) = avx2_interleave_rgb($j2, $j1, $j0);
            }
            ImageConfiguration::Bgra => {
                (row0, row1, _, _) = avx2_interleave_rgba_epi8($j2, $j1, $j0, $j3)
            }
        };
        _mm256_storeu_si256($ptr as *mut __m256i, row0);
        _mm256_storeu_si256($ptr.add(32) as *mut __m256i, row1);
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v4_quarter_u8 {
    ($ptr: expr, $configuration: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        let row0;
        match $configuration {
            ImageConfiguration::Rgba => {
                (row0, _, _, _) = avx2_interleave_rgba_epi8($j0, $j1, $j2, $j3)
            }
            ImageConfiguration::Rgb => {
                (row0, _, _) = avx2_interleave_rgb($j0, $j1, $j2);
            }
            ImageConfiguration::Bgr => {
                (row0, _, _) = avx2_interleave_rgb($j2, $j1, $j0);
            }
            ImageConfiguration::Bgra => {
                (row0, _, _, _) = avx2_interleave_rgba_epi8($j2, $j1, $j0, $j3)
            }
        };
        _mm256_storeu_si256($ptr as *mut __m256i, row0);
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v3_u8 {
    ($ptr: expr, $configuration: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        let store_rows = match $configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                avx2_interleave_rgb($j0, $j1, $j2)
            }
            ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                avx2_interleave_rgb($j2, $j1, $j0)
            }
        };
        _mm256_storeu_si256($ptr as *mut __m256i, store_rows.0);
        _mm256_storeu_si256($ptr.add(32) as *mut __m256i, store_rows.1);
        _mm256_storeu_si256($ptr.add(64) as *mut __m256i, store_rows.2);
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v3_half_u8 {
    ($ptr: expr, $configuration: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        let store_rows = match $configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                avx2_interleave_rgb($j0, $j1, $j2)
            }
            ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                avx2_interleave_rgb($j2, $j1, $j0)
            }
        };
        _mm256_storeu_si256($ptr as *mut __m256i, store_rows.0);
        _mm_storeu_si128(
            $ptr.add(32) as *mut __m128i,
            _mm256_castsi256_si128(store_rows.1),
        );
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v4_direct_f32 {
    ($ptr: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        let (v0, v1, v2, v3) = avx2_interleave_rgba_ps($j0, $j1, $j2, $j3);
        _mm256_storeu_ps($ptr, v0);
        _mm256_storeu_ps($ptr.add(8), v1);
        _mm256_storeu_ps($ptr.add(16), v2);
        _mm256_storeu_ps($ptr.add(24), v3);
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v4_f32 {
    ($ptr: expr, $image_configuration: expr, $j0: expr, $j1: expr, $j2: expr, $j3: expr) => {{
        match $image_configuration {
            ImageConfiguration::Rgba => {
                let (v0, v1, v2, v3) = avx2_interleave_rgba_ps($j0, $j1, $j2, $j3);
                _mm256_storeu_ps($ptr, v0);
                _mm256_storeu_ps($ptr.add(8), v1);
                _mm256_storeu_ps($ptr.add(16), v2);
                _mm256_storeu_ps($ptr.add(24), v3);
            }
            ImageConfiguration::Bgra => {
                let (v0, v1, v2, v3) = avx2_interleave_rgba_ps($j2, $j1, $j0, $j3);
                _mm256_storeu_ps($ptr, v0);
                _mm256_storeu_ps($ptr.add(8), v1);
                _mm256_storeu_ps($ptr.add(16), v2);
                _mm256_storeu_ps($ptr.add(24), v3);
            }
            ImageConfiguration::Rgb => {
                let (v0, v1, v2) = avx2_interleave_rgb_ps($j0, $j1, $j2);
                _mm256_storeu_ps($ptr, v0);
                _mm256_storeu_ps($ptr.add(8), v1);
                _mm256_storeu_ps($ptr.add(16), v2);
            }
            ImageConfiguration::Bgr => {
                let (v0, v1, v2) = avx2_interleave_rgb_ps($j2, $j1, $j0);
                _mm256_storeu_ps($ptr, v0);
                _mm256_storeu_ps($ptr.add(8), v1);
                _mm256_storeu_ps($ptr.add(16), v2);
            }
        }
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v3_f32 {
    ($ptr: expr, $image_configuration: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        match $image_configuration {
            ImageConfiguration::Rgba => {
                let (v0, v1, v2, v3) = avx2_interleave_rgba_ps($j0, $j1, $j2, _mm256_setzero_ps());
                _mm256_storeu_ps($ptr, v0);
                _mm256_storeu_ps($ptr.add(8), v1);
                _mm256_storeu_ps($ptr.add(16), v2);
                _mm256_storeu_ps($ptr.add(24), v3);
            }
            ImageConfiguration::Bgra => {
                let (v0, v1, v2, v3) = avx2_interleave_rgba_ps($j2, $j1, $j0, _mm256_setzero_ps());
                _mm256_storeu_ps($ptr, v0);
                _mm256_storeu_ps($ptr.add(8), v1);
                _mm256_storeu_ps($ptr.add(16), v2);
                _mm256_storeu_ps($ptr.add(24), v3);
            }
            ImageConfiguration::Rgb => {
                let (v0, v1, v2) = avx2_interleave_rgb_ps($j0, $j1, $j2);
                _mm256_storeu_ps($ptr, v0);
                _mm256_storeu_ps($ptr.add(8), v1);
                _mm256_storeu_ps($ptr.add(16), v2);
            }
            ImageConfiguration::Bgr => {
                let (v0, v1, v2) = avx2_interleave_rgb_ps($j2, $j1, $j0);
                _mm256_storeu_ps($ptr, v0);
                _mm256_storeu_ps($ptr.add(8), v1);
                _mm256_storeu_ps($ptr.add(16), v2);
            }
        }
    }};
}

#[macro_export]
macro_rules! avx_store_and_interleave_v3_direct_f32 {
    ($ptr: expr, $j0: expr, $j1: expr, $j2: expr) => {{
        let (v0, v1, v2) = avx2_interleave_rgb_ps($j0, $j1, $j2);
        _mm256_storeu_ps($ptr, v0);
        _mm256_storeu_ps($ptr.add(8), v1);
        _mm256_storeu_ps($ptr.add(16), v2);
    }};
}
