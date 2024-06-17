use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::sse::cie::{sse_triple_to_lab, sse_triple_to_lch, sse_triple_to_luv, sse_triple_to_xyz};
use crate::sse::*;
use crate::xyz_target::XyzTarget;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn sse_channels_to_xyza_laba<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> usize {
    const CHANNELS: usize = 4;
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    if !image_configuration.has_alpha() {
        panic!("Null alpha channel with requirements of linearized alpha if not supported");
    }
    let mut cx = start_cx;

    let transfer = get_sse_linear_transfer(transfer_function);

    let cq1 = _mm_set1_ps(matrix[0][0]);
    let cq2 = _mm_set1_ps(matrix[0][1]);
    let cq3 = _mm_set1_ps(matrix[0][2]);
    let cq4 = _mm_set1_ps(matrix[1][0]);
    let cq5 = _mm_set1_ps(matrix[1][1]);
    let cq6 = _mm_set1_ps(matrix[1][2]);
    let cq7 = _mm_set1_ps(matrix[2][0]);
    let cq8 = _mm_set1_ps(matrix[2][1]);
    let cq9 = _mm_set1_ps(matrix[2][2]);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 16 < width as usize {
        let (r_chan, g_chan, b_chan, a_chan);
        let src_ptr = src.add(src_offset + cx * channels);
        let row1 = _mm_loadu_si128(src_ptr as *const __m128i);
        let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
        let row3 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (c1, c2, c3) = sse_deinterleave_rgb(row1, row2, row3);
                if image_configuration == ImageConfiguration::Rgb {
                    r_chan = c1;
                    g_chan = c2;
                    b_chan = c3;
                } else {
                    r_chan = c3;
                    g_chan = c2;
                    b_chan = c1;
                }
                a_chan = _mm_setzero_si128();
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let row4 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
                let (c1, c2, c3, c4) = sse_deinterleave_rgba(row1, row2, row3, row4);
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

        let r_low = _mm_cvtepu8_epi16(r_chan);
        let g_low = _mm_cvtepu8_epi16(g_chan);
        let b_low = _mm_cvtepu8_epi16(b_chan);

        let r_low_low = _mm_cvtepu16_epi32(r_low);
        let g_low_low = _mm_cvtepu16_epi32(g_low);
        let b_low_low = _mm_cvtepu16_epi32(b_low);

        let (mut x_low_low, mut y_low_low, mut z_low_low) = sse_triple_to_xyz(
            r_low_low, g_low_low, b_low_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9, &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = sse_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = sse_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
            XyzTarget::LCH => {
                let (l, c, h) = sse_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
        }

        let a_low = _mm_cvtepu8_epi16(a_chan);
        let u8_scale = _mm_set1_ps(1f32 / 255f32);
        let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);

        let (v0, v1, v2, v3) = sse_interleave_ps_rgba(x_low_low, y_low_low, z_low_low, a_low_low);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 12), v3);

        let r_low_high = _mm_unpackhi_epi16(r_low, _mm_setzero_si128());
        let g_low_high = _mm_unpackhi_epi16(g_low, _mm_setzero_si128());
        let b_low_high = _mm_unpackhi_epi16(b_low, _mm_setzero_si128());

        let (mut x_low_high, mut y_low_high, mut z_low_high) = sse_triple_to_xyz(
            r_low_high, g_low_high, b_low_high, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = sse_triple_to_lab(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = a;
                z_low_high = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = sse_triple_to_luv(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = u;
                z_low_high = v;
            }
            XyzTarget::LCH => {
                let (l, c, h) = sse_triple_to_lch(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = c;
                z_low_high = h;
            }
        }

        let a_low_high = _mm_mul_ps(
            _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128::<8>(a_low))),
            u8_scale,
        );

        let (v0, v1, v2, v3) =
            sse_interleave_ps_rgba(x_low_high, y_low_high, z_low_high, a_low_high);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS + 12), v3);

        let r_high = _mm_unpackhi_epi8(r_chan, _mm_setzero_si128());
        let g_high = _mm_unpackhi_epi8(g_chan, _mm_setzero_si128());
        let b_high = _mm_unpackhi_epi8(b_chan, _mm_setzero_si128());

        let r_high_low = _mm_cvtepu16_epi32(r_high);
        let g_high_low = _mm_cvtepu16_epi32(g_high);
        let b_high_low = _mm_cvtepu16_epi32(b_high);

        let (mut x_high_low, mut y_high_low, mut z_high_low) = sse_triple_to_xyz(
            r_high_low, g_high_low, b_high_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = sse_triple_to_lab(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = a;
                z_high_low = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = sse_triple_to_luv(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = u;
                z_high_low = v;
            }
            XyzTarget::LCH => {
                let (l, c, h) = sse_triple_to_lch(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = c;
                z_high_low = h;
            }
        }

        let a_high = _mm_unpackhi_epi8(a_chan, _mm_setzero_si128());

        let a_high_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_high)), u8_scale);

        let (v0, v1, v2, v3) =
            sse_interleave_ps_rgba(x_high_low, y_high_low, z_high_low, a_high_low);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 2), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 2 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 2 + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 2 + 12), v3);

        let r_high_high = _mm_unpackhi_epi16(r_high, _mm_setzero_si128());
        let g_high_high = _mm_unpackhi_epi16(g_high, _mm_setzero_si128());
        let b_high_high = _mm_unpackhi_epi16(b_high, _mm_setzero_si128());

        let (mut x_high_high, mut y_high_high, mut z_high_high) = sse_triple_to_xyz(
            r_high_high,
            g_high_high,
            b_high_high,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = sse_triple_to_lab(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = a;
                z_high_high = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = sse_triple_to_luv(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = u;
                z_high_high = v;
            }
            XyzTarget::LCH => {
                let (l, c, h) = sse_triple_to_lch(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = c;
                z_high_high = h;
            }
        }

        let a_high_high = _mm_mul_ps(
            _mm_cvtepi32_ps(_mm_unpackhi_epi16(a_high, _mm_setzero_si128())),
            u8_scale,
        );

        let (v0, v1, v2, v3) =
            sse_interleave_ps_rgba(x_high_high, y_high_high, z_high_high, a_high_high);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 3), v0);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 3 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 3 + 8), v2);
        _mm_storeu_ps(dst_ptr.add(cx * CHANNELS + 4 * CHANNELS * 3 + 12), v3);

        cx += 16;
    }

    cx
}
