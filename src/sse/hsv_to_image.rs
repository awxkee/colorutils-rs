/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
use crate::sse::color::{sse_hsl_to_rgb, sse_hsv_to_rgb};
use crate::sse::{
    sse_deinterleave_rgb_epi16, sse_deinterleave_rgba_epi16, sse_interleave_rgb,
    sse_interleave_rgba,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub unsafe fn sse_hsv_u16_to_image<
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

    let v_scale = _mm_set1_ps(scale);

    let dst_ptr = dst.add(dst_offset);
    let src_load_ptr = (src as *const u8).add(src_offset) as *const u16;

    while cx + 16 < width as usize {
        let (h_chan, s_chan, v_chan, a_chan_lo);
        let src_ptr = src_load_ptr.add(cx * channels);

        let row0 = _mm_loadu_si128(src_ptr as *const __m128i);
        let row1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
        let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);

        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (h_c, s_c, v_c) = sse_deinterleave_rgb_epi16(row0, row1, row2);
                h_chan = h_c;
                s_chan = s_c;
                v_chan = v_c;
                a_chan_lo = _mm_set1_epi16(255);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let row3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
                let (h_c, s_c, v_c, a_c) = sse_deinterleave_rgba_epi16(row0, row1, row2, row3);
                h_chan = h_c;
                s_chan = s_c;
                v_chan = v_c;
                a_chan_lo = a_c;
            }
        }

        let h_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(h_chan));
        let s_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(s_chan));
        let v_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(v_chan));

        let (r_low, g_low, b_low) = match target {
            HsvTarget::HSV => sse_hsv_to_rgb(h_low, s_low, v_low, v_scale),
            HsvTarget::HSL => sse_hsl_to_rgb(h_low, s_low, v_low, v_scale),
        };

        let zeros = _mm_setzero_si128();

        let h_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(h_chan, zeros));
        let s_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(s_chan, zeros));
        let v_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_chan, zeros));

        let (r_high, g_high, b_high) = match target {
            HsvTarget::HSV => sse_hsv_to_rgb(h_high, s_high, v_high, v_scale),
            HsvTarget::HSL => sse_hsl_to_rgb(h_high, s_high, v_high, v_scale),
        };

        let r_chan_16_lo = _mm_packus_epi32(r_low, r_high);
        let g_chan_16_lo = _mm_packus_epi32(g_low, g_high);
        let b_chan_16_lo = _mm_packus_epi32(b_low, b_high);

        let (h_chan, s_chan, v_chan, a_chan_hi);
        let src_ptr = src_load_ptr.add(cx * channels);

        let src_ptr = src_ptr.add(8 * channels);
        let row0 = _mm_loadu_si128(src_ptr as *const __m128i);
        let row1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
        let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);

        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (h_c, s_c, v_c) = sse_deinterleave_rgb_epi16(row0, row1, row2);
                h_chan = h_c;
                s_chan = s_c;
                v_chan = v_c;
                a_chan_hi = _mm_set1_epi16(255);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let row3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
                let (h_c, s_c, v_c, a_c) = sse_deinterleave_rgba_epi16(row0, row1, row2, row3);
                h_chan = h_c;
                s_chan = s_c;
                v_chan = v_c;
                a_chan_hi = a_c;
            }
        }

        let h_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(h_chan));
        let s_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(s_chan));
        let v_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(v_chan));

        let (r_low, g_low, b_low) = match target {
            HsvTarget::HSV => sse_hsv_to_rgb(h_low, s_low, v_low, v_scale),
            HsvTarget::HSL => sse_hsl_to_rgb(h_low, s_low, v_low, v_scale),
        };

        let zeros = _mm_setzero_si128();

        let h_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(h_chan, zeros));
        let s_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(s_chan, zeros));
        let v_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_chan, zeros));

        let (r_high, g_high, b_high) = match target {
            HsvTarget::HSV => sse_hsv_to_rgb(h_high, s_high, v_high, v_scale),
            HsvTarget::HSL => sse_hsl_to_rgb(h_high, s_high, v_high, v_scale),
        };

        let r_chan_16_hi = _mm_packus_epi32(r_low, r_high);
        let g_chan_16_hi = _mm_packus_epi32(g_low, g_high);
        let b_chan_16_hi = _mm_packus_epi32(b_low, b_high);

        let r_chan = _mm_packus_epi16(r_chan_16_lo, r_chan_16_hi);
        let g_chan = _mm_packus_epi16(g_chan_16_lo, g_chan_16_hi);
        let b_chan = _mm_packus_epi16(b_chan_16_lo, b_chan_16_hi);

        let ptr = dst_ptr.add(cx * channels);
        if USE_ALPHA {
            let a_chan = _mm_packus_epi16(a_chan_lo, a_chan_hi);
            let (rgba0, rgba1, rgba2, rgba3) = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    sse_interleave_rgba(r_chan, g_chan, b_chan, a_chan)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    sse_interleave_rgba(b_chan, g_chan, r_chan, a_chan)
                }
            };
            _mm_storeu_si128(ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(ptr.add(16) as *mut __m128i, rgba1);
            _mm_storeu_si128(ptr.add(32) as *mut __m128i, rgba2);
            _mm_storeu_si128(ptr.add(48) as *mut __m128i, rgba3);
        } else {
            let (rgba0, rgba1, rgba2) = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    sse_interleave_rgb(r_chan, g_chan, b_chan)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    sse_interleave_rgb(b_chan, g_chan, r_chan)
                }
            };
            _mm_storeu_si128(ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(ptr.add(16) as *mut __m128i, rgba1);
            _mm_storeu_si128(ptr.add(32) as *mut __m128i, rgba2);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let (h_chan, s_chan, v_chan, a_chan_lo);
        let src_ptr = src_load_ptr.add(cx * channels);

        let row0 = _mm_loadu_si128(src_ptr as *const __m128i);
        let row1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
        let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);

        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (h_c, s_c, v_c) = sse_deinterleave_rgb_epi16(row0, row1, row2);
                h_chan = h_c;
                s_chan = s_c;
                v_chan = v_c;
                a_chan_lo = _mm_set1_epi16(255);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let row3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
                let (h_c, s_c, v_c, a_c) = sse_deinterleave_rgba_epi16(row0, row1, row2, row3);
                h_chan = h_c;
                s_chan = s_c;
                v_chan = v_c;
                a_chan_lo = a_c;
            }
        }

        let h_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(h_chan));
        let s_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(s_chan));
        let v_low = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(v_chan));

        let (r_low, g_low, b_low) = match target {
            HsvTarget::HSV => sse_hsv_to_rgb(h_low, s_low, v_low, v_scale),
            HsvTarget::HSL => sse_hsl_to_rgb(h_low, s_low, v_low, v_scale),
        };

        let zeros = _mm_setzero_si128();

        let h_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(h_chan, zeros));
        let s_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(s_chan, zeros));
        let v_high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_chan, zeros));

        let (r_high, g_high, b_high) = match target {
            HsvTarget::HSV => sse_hsv_to_rgb(h_high, s_high, v_high, v_scale),
            HsvTarget::HSL => sse_hsl_to_rgb(h_high, s_high, v_high, v_scale),
        };

        let r_chan_16_lo = _mm_packus_epi32(r_low, r_high);
        let g_chan_16_lo = _mm_packus_epi32(g_low, g_high);
        let b_chan_16_lo = _mm_packus_epi32(b_low, b_high);

        let r_chan = _mm_packus_epi16(r_chan_16_lo, zeros);
        let g_chan = _mm_packus_epi16(g_chan_16_lo, zeros);
        let b_chan = _mm_packus_epi16(b_chan_16_lo, zeros);

        let ptr = dst_ptr.add(cx * channels);
        if USE_ALPHA {
            let a_chan = _mm_packus_epi16(a_chan_lo, _mm_setzero_si128());
            let (rgba0, rgba1, _, _) = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    sse_interleave_rgba(r_chan, g_chan, b_chan, a_chan)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    sse_interleave_rgba(b_chan, g_chan, r_chan, a_chan)
                }
            };
            _mm_storeu_si128(ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(ptr.add(16) as *mut __m128i, rgba1);
        } else {
            let (rgba0, rgba1, _) = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    sse_interleave_rgb(r_chan, g_chan, b_chan)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    sse_interleave_rgb(b_chan, g_chan, r_chan)
                }
            };
            _mm_storeu_si128(ptr as *mut __m128i, rgba0);
            std::ptr::copy_nonoverlapping(&rgba1 as *const _ as *const u8, ptr.add(16), 8);
        }

        cx += 8;
    }

    cx
}
