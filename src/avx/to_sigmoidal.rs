use crate::avx::sigmoidal::avx_rgb_to_sigmoidal;
use crate::avx::{
    avx2_deinterleave_rgb_epi8, avx2_deinterleave_rgba_epi8, avx2_interleave_rgb_ps,
    avx2_interleave_rgba_ps,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::image::ImageConfiguration;

#[inline]
pub unsafe fn avx_image_to_sigmoidal_row<
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

    while cx + 32 < width as usize {
        let (r_chan, g_chan, b_chan, a_chan);
        let src_ptr = src.add(cx * channels);
        let row1 = _mm256_loadu_si256(src_ptr as *const __m256i);
        let row2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
        let row3 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (rgb0_, rgb1_, rgb2_) = avx2_deinterleave_rgb_epi8(row1, row2, row3);
                if image_configuration == ImageConfiguration::Rgb {
                    r_chan = rgb0_;
                    g_chan = rgb1_;
                    b_chan = rgb2_;
                } else {
                    r_chan = rgb2_;
                    g_chan = rgb1_;
                    b_chan = rgb0_;
                }
                a_chan = _mm256_setzero_si256();
            }
            ImageConfiguration::Rgba => {
                let row4 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
                let (rgb0_, rgb1_, rgb2_, rgb3_) =
                    avx2_deinterleave_rgba_epi8(row1, row2, row3, row4);
                r_chan = rgb0_;
                g_chan = rgb1_;
                b_chan = rgb2_;
                a_chan = rgb3_;
            }
            ImageConfiguration::Bgra => {
                let row4 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
                let (rgb0_, rgb1_, rgb2_, rgb3_) =
                    avx2_deinterleave_rgba_epi8(row1, row2, row3, row4);
                r_chan = rgb2_;
                g_chan = rgb1_;
                b_chan = rgb0_;
                a_chan = rgb3_;
            }
        }

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));
        let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (x_low_low, y_low_low, z_low_low) =
            avx_rgb_to_sigmoidal(r_low_low, g_low_low, b_low_low);

        let u8_scale = _mm256_set1_ps(1f32 / 255f32);
        let a_low_low = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
            u8_scale,
        );

        if USE_ALPHA {
            let (v0, v1, v2, v3) =
                avx2_interleave_rgba_ps(x_low_low, y_low_low, z_low_low, a_low_low);
            _mm256_storeu_ps(dst_ptr.add(cx * channels), v0);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8), v1);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 16), v2);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 24), v3);
        } else {
            let (v0, v1, v2) = avx2_interleave_rgb_ps(x_low_low, y_low_low, z_low_low);
            _mm256_storeu_ps(dst_ptr.add(cx * channels), v0);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8), v1);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 16), v2);
        }

        let r_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r_low));
        let g_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(g_low));
        let b_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(b_low));

        let (x_low_high, y_low_high, z_low_high) =
            avx_rgb_to_sigmoidal(r_low_high, g_low_high, b_low_high);

        let a_low_high = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(a_low))),
            u8_scale,
        );

        if USE_ALPHA {
            let (v0, v1, v2, v3) =
                avx2_interleave_rgba_ps(x_low_high, y_low_high, z_low_high, a_low_high);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels), v0);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels + 8), v1);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels + 16), v2);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels + 24), v3);
        } else {
            let (v0, v1, v2) = avx2_interleave_rgb_ps(x_low_high, y_low_high, z_low_high);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels), v0);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels + 8), v1);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels + 16), v2);
        }

        let r_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_chan));
        let g_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_chan));
        let b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_chan));

        let r_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_high));
        let g_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_high));
        let b_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_high));

        let (x_high_low, y_high_low, z_high_low) =
            avx_rgb_to_sigmoidal(r_high_low, g_high_low, b_high_low);

        let a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a_chan));

        if USE_ALPHA {
            let a_high_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_high))),
                u8_scale,
            );
            let (v0, v1, v2, v3) =
                avx2_interleave_rgba_ps(x_high_low, y_high_low, z_high_low, a_high_low);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 2), v0);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 2 + 8), v1);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 2 + 16), v2);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 2 + 24), v3);
        } else {
            let (v0, v1, v2) = avx2_interleave_rgb_ps(x_high_low, y_high_low, z_high_low);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 2), v0);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 2 + 8), v1);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 2 + 16), v2);
        }

        let r_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r_high));
        let g_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(g_high));
        let b_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(b_high));

        let (x_high_high, y_high_high, z_high_high) =
            avx_rgb_to_sigmoidal(r_high_high, g_high_high, b_high_high);

        if USE_ALPHA {
            let a_high_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(a_high))),
                u8_scale,
            );

            let (v0, v1, v2, v3) =
                avx2_interleave_rgba_ps(x_high_high, y_high_high, z_high_high, a_high_high);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 3), v0);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 3 + 8), v1);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 3 + 16), v2);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 3 + 24), v3);
        } else {
            let (v0, v1, v2) = avx2_interleave_rgb_ps(x_high_high, y_high_high, z_high_high);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 3), v0);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 3 + 8), v1);
            _mm256_storeu_ps(dst_ptr.add(cx * channels + 8 * channels * 3 + 16), v2);
        }

        cx += 32;
    }

    cx
}
