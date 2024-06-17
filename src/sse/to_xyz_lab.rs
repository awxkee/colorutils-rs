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
pub unsafe fn sse_channels_to_xyz_or_lab<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    a_linearized: *mut f32,
    a_offset: usize,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> usize {
    if USE_ALPHA {
        if a_linearized.is_null() {
            panic!("Null alpha channel with requirements of linearized alpha if not supported");
        }
    }
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
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
                a_chan = _mm_set1_epi8(0);
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

        let (v0, v1, v2) = sse_interleave_ps_rgb(x_low_low, y_low_low, z_low_low);
        _mm_storeu_ps(dst_ptr.add(cx * 3), v0);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 8), v2);

        let r_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(r_low));
        let g_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(g_low));
        let b_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b_low));

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

        let (v0, v1, v2) = sse_interleave_ps_rgb(x_low_high, y_low_high, z_low_high);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3), v0);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 + 8), v2);

        let r_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(r_chan));
        let g_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(g_chan));
        let b_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(b_chan));

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

        let (v0, v1, v2) = sse_interleave_ps_rgb(x_high_low, y_high_low, z_high_low);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 2), v0);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 2 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 2 + 8), v2);

        let r_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(r_high));
        let g_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(g_high));
        let b_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b_high));

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

        let (v0, v1, v2) = sse_interleave_ps_rgb(x_high_high, y_high_high, z_high_high);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 3), v0);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 3 + 4), v1);
        _mm_storeu_ps(dst_ptr.add(cx * 3 + 4 * 3 * 3 + 8), v2);

        if USE_ALPHA {
            let a_ptr = (a_linearized as *mut u8).add(a_offset) as *mut f32;

            let a_low = _mm_cvtepu8_epi16(a_chan);

            let u8_scale = _mm_set1_ps(1f32 / 255f32);

            let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);

            _mm_storeu_ps(a_ptr.add(cx), a_low_low);

            let a_low_high = _mm_mul_ps(
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128::<8>(a_low))),
                u8_scale,
            );

            _mm_storeu_ps(a_ptr.add(cx + 4), a_low_high);

            let a_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(a_chan));

            let a_high_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_high)), u8_scale);

            _mm_storeu_ps(a_ptr.add(cx + 4 * 2), a_high_low);

            let a_high_high = _mm_mul_ps(
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128::<8>(a_high))),
                u8_scale,
            );

            _mm_storeu_ps(a_ptr.add(cx + 4 * 3), a_high_high);
        }

        cx += 16;
    }

    cx
}
