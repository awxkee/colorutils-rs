use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
use crate::sse::sse_color::{sse_rgb_to_hsl, sse_rgb_to_hsv};
use crate::sse::{
    sse_deinterleave_rgb, sse_deinterleave_rgba, sse_interleave_rgb_epi16,
    sse_interleave_rgba_epi16,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub unsafe fn sse_channels_to_hsv_u16<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut u16,
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

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut u16;

    while cx + 16 < width as usize {
        let (r_chan, g_chan, b_chan, a_chan);
        let src_ptr = src.add(src_offset + cx * channels);
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

        let r_low_low = _mm_unpacklo_epi16(r_low, zeros);
        let g_low_low = _mm_unpacklo_epi16(g_low, zeros);
        let b_low_low = _mm_unpacklo_epi16(b_low, zeros);

        let (x_low_low, y_low_low, z_low_low) = match target {
            HsvTarget::HSV => sse_rgb_to_hsv(r_low_low, g_low_low, b_low_low, v_scale),
            HsvTarget::HSL => sse_rgb_to_hsl(r_low_low, g_low_low, b_low_low, v_scale),
        };

        let a_low = _mm_unpacklo_epi8(a_chan, zeros);

        let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) = match target {
            HsvTarget::HSV => sse_rgb_to_hsv(r_low_high, g_low_high, b_low_high, v_scale),
            HsvTarget::HSL => sse_rgb_to_hsl(r_low_high, g_low_high, b_low_high, v_scale),
        };

        const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
        let x_low = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(x_low_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(x_low_high)),
        );
        let y_low = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(y_low_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(y_low_high)),
        );
        let z_low = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(z_low_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(z_low_high)),
        );

        if USE_ALPHA {
            let (row1, row2, row3, row4) = sse_interleave_rgba_epi16(x_low, y_low, z_low, a_low);
            let ptr = dst_ptr.add(cx * channels);
            _mm_storeu_si128(ptr as *mut __m128i, row1);
            _mm_storeu_si128(ptr.add(8) as *mut __m128i, row2);
            _mm_storeu_si128(ptr.add(16) as *mut __m128i, row3);
            _mm_storeu_si128(ptr.add(24) as *mut __m128i, row4);
        } else {
            let (row1, row2, row3) = sse_interleave_rgb_epi16(x_low, y_low, z_low);
            let ptr = dst_ptr.add(cx * channels);
            _mm_storeu_si128(ptr as *mut __m128i, row1);
            _mm_storeu_si128(ptr.add(8) as *mut __m128i, row2);
            _mm_storeu_si128(ptr.add(16) as *mut __m128i, row3);
        }

        let r_high = _mm_unpackhi_epi8(r_chan, zeros);
        let g_high = _mm_unpackhi_epi8(g_chan, zeros);
        let b_high = _mm_unpackhi_epi8(b_chan, zeros);

        let r_high_low = _mm_unpacklo_epi16(r_high, zeros);
        let g_high_low = _mm_unpacklo_epi16(g_high, zeros);
        let b_high_low = _mm_unpacklo_epi16(b_high, zeros);

        let (x_high_low, y_high_low, z_high_low) = match target {
            HsvTarget::HSV => sse_rgb_to_hsv(r_high_low, g_high_low, b_high_low, v_scale),
            HsvTarget::HSL => sse_rgb_to_hsl(r_high_low, g_high_low, b_high_low, v_scale),
        };

        let a_high = _mm_unpackhi_epi8(a_chan, zeros);

        let r_high_high = _mm_unpackhi_epi16(r_high, zeros);
        let g_high_high = _mm_unpackhi_epi16(g_high, zeros);
        let b_high_high = _mm_unpackhi_epi16(b_high, zeros);

        let (x_high_high, y_high_high, z_high_high) = match target {
            HsvTarget::HSV => sse_rgb_to_hsv(r_high_high, g_high_high, b_high_high, v_scale),
            HsvTarget::HSL => sse_rgb_to_hsl(r_high_high, g_high_high, b_high_high, v_scale),
        };

        let x_high = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(x_high_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(x_high_high)),
        );
        let y_high = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(y_high_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(y_high_high)),
        );
        let z_high = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(z_high_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(z_high_high)),
        );

        if USE_ALPHA {
            let (row1, row2, row3, row4) =
                sse_interleave_rgba_epi16(x_high, y_high, z_high, a_high);
            let ptr = dst_ptr.add(cx * channels + 8 * channels);
            _mm_storeu_si128(ptr as *mut __m128i, row1);
            _mm_storeu_si128(ptr.add(8) as *mut __m128i, row2);
            _mm_storeu_si128(ptr.add(16) as *mut __m128i, row3);
            _mm_storeu_si128(ptr.add(24) as *mut __m128i, row4);
        } else {
            let (row1, row2, row3) = sse_interleave_rgb_epi16(x_high, y_high, z_high);
            let ptr = dst_ptr.add(cx * channels + 8 * channels);
            _mm_storeu_si128(ptr as *mut __m128i, row1);
            _mm_storeu_si128(ptr.add(8) as *mut __m128i, row2);
            _mm_storeu_si128(ptr.add(16) as *mut __m128i, row3);
        }

        cx += 16;
    }

    cx
}
