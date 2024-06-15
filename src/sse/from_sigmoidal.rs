#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::image::ImageConfiguration;
use crate::sse::sigmoidal::sse_sigmoidal_to_rgb;
use crate::sse::{
    sse_deinterleave_rgb_ps, sse_deinterleave_rgba_ps, sse_interleave_rgb, sse_interleave_rgba,
};

#[inline(always)]
unsafe fn vld_sigmoidal<const CHANNELS_CONFIGURATION: u8>(
    src: *const f32,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let v_scale_color = _mm_set1_ps(255f32);
    let pixel_0 = _mm_loadu_ps(src);
    let pixel_1 = _mm_loadu_ps(src.add(4));
    let pixel_2 = _mm_loadu_ps(src.add(8));
    if image_configuration.has_alpha() {
        let pixel_3 = _mm_loadu_ps(src.add(12));
        let (sr, sg, sb, sa) = sse_deinterleave_rgba_ps(pixel_0, pixel_1, pixel_2, pixel_3);

        let (r, g, b) = sse_sigmoidal_to_rgb(sr, sg, sb);
        const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
        let a_f32 = _mm_mul_ps(sa, v_scale_color);
        (
            r,
            g,
            b,
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(a_f32)),
        )
    } else {
        let (sr, sg, sb) = sse_deinterleave_rgb_ps(pixel_0, pixel_1, pixel_2);

        let (r, g, b) = sse_sigmoidal_to_rgb(sr, sg, sb);
        (r, g, b, _mm_setzero_si128())
    }
}

#[inline(always)]
pub unsafe fn sse_from_sigmoidal_row<const CHANNELS_CONFIGURATION: u8>(
    start_cx: usize,
    src: *const f32,
    dst: *mut u8,
    width: u32,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let channels = image_configuration.get_channels_count();

    let mut cx = start_cx;

    while cx + 16 < width as usize {
        let offset_src_ptr = src.add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            vld_sigmoidal::<CHANNELS_CONFIGURATION>(src_ptr_0);

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            vld_sigmoidal::<CHANNELS_CONFIGURATION>(src_ptr_1);

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            vld_sigmoidal::<CHANNELS_CONFIGURATION>(src_ptr_2);

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            vld_sigmoidal::<CHANNELS_CONFIGURATION>(src_ptr_3);

        let r_row01 = _mm_packs_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packs_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packs_epi32(b_row0_, b_row1_);
        let a_row01 = _mm_packs_epi32(a_row0_, a_row1_);

        let r_row23 = _mm_packs_epi32(r_row2_, r_row3_);
        let g_row23 = _mm_packs_epi32(g_row2_, g_row3_);
        let b_row23 = _mm_packs_epi32(b_row2_, b_row3_);
        let a_row23 = _mm_packs_epi32(a_row2_, a_row3_);

        let r_row = _mm_packus_epi16(r_row01, r_row23);
        let g_row = _mm_packus_epi16(g_row01, g_row23);
        let b_row = _mm_packus_epi16(b_row01, b_row23);
        let a_row = _mm_packus_epi16(a_row01, a_row23);

        let dst_ptr = dst.add(cx * channels);

        match image_configuration {
            ImageConfiguration::Rgb => {
                let (rgb0, rgb1, rgb2) = sse_interleave_rgb(r_row, g_row, b_row);
                _mm_storeu_si128(dst_ptr as *mut __m128i, rgb0);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgb1);
                _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgb2);
            }
            ImageConfiguration::Rgba => {
                let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba(r_row, g_row, b_row, a_row);
                _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
                _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
                _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);
            }
            ImageConfiguration::Bgra => {
                let (bgra0, bgra1, bgra2, bgra3) = sse_interleave_rgba(b_row, g_row, r_row, a_row);
                _mm_storeu_si128(dst_ptr as *mut __m128i, bgra0);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, bgra1);
                _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, bgra2);
                _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, bgra3);
            }
            ImageConfiguration::Bgr => {
                let (bgr0, bgr1, bgr2) = sse_interleave_rgb(b_row, g_row, r_row);
                _mm_storeu_si128(dst_ptr as *mut __m128i, bgr0);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, bgr1);
                _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, bgr2);
            }
        }
        cx += 16;
    }

    cx
}
