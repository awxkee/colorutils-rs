#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::image::ImageConfiguration;
use crate::sse::sigmoidal::sse_rgb_to_sigmoidal;
use crate::sse::{
    sse_deinterleave_rgb, sse_deinterleave_rgba, sse_interleave_ps_rgb, sse_interleave_ps_rgba,
};

#[inline]
pub unsafe fn sse_image_to_sigmoidal_row<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
>(
    start_cx: usize,
    src: *const u8,
    width: u32,
    dst: *mut f32,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let mut cx = start_cx;
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Use alpha flag used on image without alpha");
        }
    }

    let channels = image_configuration.get_channels_count();

    let dst_ptr = (dst as *mut u8) as *mut f32;

    while cx + 16 < width as usize {
        let (r_chan, g_chan, b_chan, a_chan);
        let src_ptr = src.add(cx * channels);
        let row1 = _mm_loadu_si128(src_ptr as *const __m128i);
        let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
        let row3 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (rgb0_, rgb1_, rgb2_) = sse_deinterleave_rgb(row1, row2, row3);
                if image_configuration == ImageConfiguration::Rgb {
                    r_chan = rgb0_;
                    g_chan = rgb1_;
                    b_chan = rgb2_;
                } else {
                    r_chan = rgb2_;
                    g_chan = rgb1_;
                    b_chan = rgb0_;
                }
                a_chan = _mm_setzero_si128();
            }
            ImageConfiguration::Rgba => {
                let row4 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
                let (rgb0_, rgb1_, rgb2_, rgb3_) = sse_deinterleave_rgba(row1, row2, row3, row4);
                r_chan = rgb0_;
                g_chan = rgb1_;
                b_chan = rgb2_;
                a_chan = rgb3_;
            }
            ImageConfiguration::Bgra => {
                let row4 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
                let (rgb0_, rgb1_, rgb2_, rgb3_) = sse_deinterleave_rgba(row1, row2, row3, row4);
                r_chan = rgb2_;
                g_chan = rgb1_;
                b_chan = rgb0_;
                a_chan = rgb3_;
            }
        }

        let zeros = _mm_setzero_si128();

        let r_low = _mm_unpacklo_epi8(r_chan, zeros);
        let g_low = _mm_unpacklo_epi8(g_chan, zeros);
        let b_low = _mm_unpacklo_epi8(b_chan, zeros);
        let a_low = _mm_cvtepu8_epi16(a_chan);

        let r_low_low = _mm_unpacklo_epi16(r_low, zeros);
        let g_low_low = _mm_unpacklo_epi16(g_low, zeros);
        let b_low_low = _mm_unpacklo_epi16(b_low, zeros);

        let (x_low_low, y_low_low, z_low_low) =
            sse_rgb_to_sigmoidal(r_low_low, g_low_low, b_low_low);

        let u8_scale = _mm_set1_ps(1f32 / 255f32);
        let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);

        if USE_ALPHA {
            let (v0, v1, v2, v3) =
                sse_interleave_ps_rgba(x_low_low, y_low_low, z_low_low, a_low_low);
            _mm_storeu_ps(dst_ptr.add(cx * channels), v0);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 8), v2);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 12), v3);
        } else {
            let (v0, v1, v2) = sse_interleave_ps_rgb(x_low_low, y_low_low, z_low_low);
            _mm_storeu_ps(dst_ptr.add(cx * channels), v0);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 8), v2);
        }

        let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) =
            sse_rgb_to_sigmoidal(r_low_high, g_low_high, b_low_high);

        let a_low_high = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_low, zeros)), u8_scale);

        if USE_ALPHA {
            let (v0, v1, v2, v3) =
                sse_interleave_ps_rgba(x_low_high, y_low_high, z_low_high, a_low_high);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels), v0);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels + 8), v2);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels + 12), v3);
        } else {
            let (v0, v1, v2) = sse_interleave_ps_rgb(x_low_high, y_low_high, z_low_high);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels), v0);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels + 8), v2);
        }

        let r_high = _mm_unpackhi_epi8(r_chan, zeros);
        let g_high = _mm_unpackhi_epi8(g_chan, zeros);
        let b_high = _mm_unpackhi_epi8(b_chan, zeros);

        let r_high_low = _mm_unpacklo_epi16(r_high, zeros);
        let g_high_low = _mm_unpacklo_epi16(g_high, zeros);
        let b_high_low = _mm_unpacklo_epi16(b_high, zeros);

        let (x_high_low, y_high_low, z_high_low) =
            sse_rgb_to_sigmoidal(r_high_low, g_high_low, b_high_low);

        let a_high = _mm_unpackhi_epi8(a_chan, zeros);

        if USE_ALPHA {
            let a_high_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_high)), u8_scale);
            let (v0, v1, v2, v3) =
                sse_interleave_ps_rgba(x_high_low, y_high_low, z_high_low, a_high_low);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 2), v0);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 2 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 2 + 8), v2);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 2 + 12), v3);
        } else {
            let (v0, v1, v2) = sse_interleave_ps_rgb(x_high_low, y_high_low, z_high_low);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 2), v0);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 2 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 2 + 8), v2);
        }

        let r_high_high = _mm_unpackhi_epi16(r_high, zeros);
        let g_high_high = _mm_unpackhi_epi16(g_high, zeros);
        let b_high_high = _mm_unpackhi_epi16(b_high, zeros);

        let (x_high_high, y_high_high, z_high_high) =
            sse_rgb_to_sigmoidal(r_high_high, g_high_high, b_high_high);

        if USE_ALPHA {
            let a_high_high = _mm_mul_ps(
                _mm_cvtepi32_ps(_mm_unpackhi_epi16(a_high, _mm_setzero_si128())),
                u8_scale,
            );

            let (v0, v1, v2, v3) =
                sse_interleave_ps_rgba(x_high_high, y_high_high, z_high_high, a_high_high);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 3), v0);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 3 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 3 + 8), v2);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 3 + 12), v3);
        } else {
            let (v0, v1, v2) = sse_interleave_ps_rgb(x_high_high, y_high_high, z_high_high);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 3), v0);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 3 + 4), v1);
            _mm_storeu_ps(dst_ptr.add(cx * channels + 4 * channels * 3 + 8), v2);
        }

        cx += 16;
    }

    cx
}
