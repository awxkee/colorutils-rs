#[allow(unused_imports)]
use crate::image::ImageConfiguration;
#[allow(unused_imports)]
use crate::sse::*;
#[allow(unused_imports)]
use crate::TransferFunction;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
unsafe fn sse_gamma_vld<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: *const f32,
    transfer_function: TransferFunction,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let d_alpha = _mm_set1_ps(1f32);
    let transfer = get_sse_gamma_transfer(transfer_function);
    let v_scale_alpha = _mm_set1_ps(255f32);
    let (mut r_f32, mut g_f32, mut b_f32, mut a_f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let row0 = _mm_loadu_ps(src);
    let row1 = _mm_loadu_ps(src.add(4));
    let row2 = _mm_loadu_ps(src.add(8));

    match image_configuration {
        ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
            let row3 = _mm_loadu_ps(src.add(12));
            let (v0, v1, v2, v3) = sse_deinterleave_rgba_ps(row0, row1, row2, row3);
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
            let rgb_pixels = sse_deinterleave_rgb_ps(row0, row1, row2);
            if image_configuration == ImageConfiguration::Rgb {
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

    r_f32 = transfer(r_f32);
    g_f32 = transfer(g_f32);
    b_f32 = transfer(b_f32);
    r_f32 = _mm_mul_ps(r_f32, v_scale_alpha);
    g_f32 = _mm_mul_ps(g_f32, v_scale_alpha);
    b_f32 = _mm_mul_ps(b_f32, v_scale_alpha);
    if USE_ALPHA {
        a_f32 = _mm_mul_ps(a_f32, v_scale_alpha);
    }
    (
        _mm_cvtps_epi32(r_f32),
        _mm_cvtps_epi32(g_f32),
        _mm_cvtps_epi32(b_f32),
        _mm_cvtps_epi32(a_f32),
    )
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub unsafe fn sse_linear_to_gamma<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, transfer_function);

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_1, transfer_function);

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_2, transfer_function);

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            sse_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_3, transfer_function);

        let r_row01 = _mm_packus_epi32(r_row0_, r_row1_);
        let g_row01 = _mm_packus_epi32(g_row0_, g_row1_);
        let b_row01 = _mm_packus_epi32(b_row0_, b_row1_);

        let r_row23 = _mm_packus_epi32(r_row2_, r_row3_);
        let g_row23 = _mm_packus_epi32(g_row2_, g_row3_);
        let b_row23 = _mm_packus_epi32(b_row2_, b_row3_);

        let r_row = _mm_packus_epi16(r_row01, r_row23);
        let g_row = _mm_packus_epi16(g_row01, g_row23);
        let b_row = _mm_packus_epi16(b_row01, b_row23);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
            let a_row01 = _mm_packus_epi32(a_row0_, a_row1_);
            let a_row23 = _mm_packus_epi32(a_row2_, a_row3_);
            let a_row = _mm_packus_epi16(a_row01, a_row23);
            let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba(r_row, g_row, b_row, a_row);
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
            _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
            _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);
        } else {
            let (rgb0, rgb1, rgb2) = sse_interleave_rgb(r_row, g_row, b_row);
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgb0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgb1);
            _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgb2);
        }

        cx += 16;
    }

    cx
}
