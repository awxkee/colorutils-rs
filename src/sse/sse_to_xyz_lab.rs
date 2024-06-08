use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::luv::{LUV_CUTOFF_FORWARD_Y, LUV_MULTIPLIER_FORWARD_Y};
#[allow(unused_imports)]
use crate::sse::*;
#[allow(unused_imports)]
use crate::image_to_xyz_lab::XyzTarget;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn sse_triple_to_xyz(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    c1: __m128,
    c2: __m128,
    c3: __m128,
    c4: __m128,
    c5: __m128,
    c6: __m128,
    c7: __m128,
    c8: __m128,
    c9: __m128,
    transfer: &unsafe fn(__m128) -> __m128,
) -> (__m128, __m128, __m128) {
    let u8_scale = _mm_set1_ps(1f32 / 255f32);
    let r_f = _mm_mul_ps(_mm_cvtepi32_ps(r), u8_scale);
    let g_f = _mm_mul_ps(_mm_cvtepi32_ps(g), u8_scale);
    let b_f = _mm_mul_ps(_mm_cvtepi32_ps(b), u8_scale);
    let r_linear = transfer(r_f);
    let g_linear = transfer(g_f);
    let b_linear = transfer(b_f);

    let (x, y, z) = _mm_color_matrix_ps(
        r_linear, g_linear, b_linear, c1, c2, c3, c4, c5, c6, c7, c8, c9,
    );
    (x, y, z)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) unsafe fn sse_triple_to_luv(
    x: __m128,
    y: __m128,
    z: __m128,
) -> (__m128, __m128, __m128) {
    let zeros = _mm_setzero_ps();
    let den = _mm_prefer_fma_ps(
        _mm_prefer_fma_ps(x, z, _mm_set1_ps(3f32)),
        y,
        _mm_set1_ps(15f32),
    );
    let nan_mask = _mm_cmpeq_ps(den, _mm_set1_ps(0f32));
    let l_low_mask = _mm_cmplt_ps(y, _mm_set1_ps(LUV_CUTOFF_FORWARD_Y));
    let y_cbrt = _mm_cbrt_ps(y);
    let l = _mm_select_ps(
        l_low_mask,
        _mm_mul_ps(y, _mm_set1_ps(LUV_MULTIPLIER_FORWARD_Y)),
        _mm_prefer_fma_ps(_mm_set1_ps(-16f32), y_cbrt, _mm_set1_ps(116f32)),
    );
    let u_prime = _mm_div_ps(_mm_mul_ps(x, _mm_set1_ps(4f32)), den);
    let v_prime = _mm_div_ps(_mm_mul_ps(y, _mm_set1_ps(9f32)), den);
    let sub_u_prime = _mm_sub_ps(u_prime, _mm_set1_ps(crate::luv::LUV_WHITE_U_PRIME));
    let sub_v_prime = _mm_sub_ps(v_prime, _mm_set1_ps(crate::luv::LUV_WHITE_V_PRIME));
    let l13 = _mm_mul_ps(l, _mm_set1_ps(13f32));
    let u = _mm_select_ps(nan_mask, zeros, _mm_mul_ps(l13, sub_u_prime));
    let v = _mm_select_ps(nan_mask, zeros, _mm_mul_ps(l13, sub_v_prime));
    (l, u, v)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) unsafe fn sse_triple_to_lab(
    x: __m128,
    y: __m128,
    z: __m128,
) -> (__m128, __m128, __m128) {
    let x = _mm_mul_ps(x, _mm_set1_ps(100f32 / 95.047f32));
    let y = _mm_mul_ps(y, _mm_set1_ps(100f32 / 100f32));
    let z = _mm_mul_ps(z, _mm_set1_ps(100f32 / 108.883f32));
    let cbrt_x = _mm_cbrt_ps(x);
    let cbrt_y = _mm_cbrt_ps(y);
    let cbrt_z = _mm_cbrt_ps(z);
    let s_1 = _mm_set1_ps(16.0 / 116.0);
    let s_2 = _mm_set1_ps(7.787);
    let lower_x = _mm_prefer_fma_ps(s_1, s_2, x);
    let lower_y = _mm_prefer_fma_ps(s_1, s_2, y);
    let lower_z = _mm_prefer_fma_ps(s_1, s_2, z);
    let cutoff = _mm_set1_ps(0.008856f32);
    let x = _mm_select_ps(_mm_cmpgt_ps(x, cutoff), cbrt_x, lower_x);
    let y = _mm_select_ps(_mm_cmpgt_ps(y, cutoff), cbrt_y, lower_y);
    let z = _mm_select_ps(_mm_cmpgt_ps(z, cutoff), cbrt_z, lower_z);
    let l = _mm_prefer_fma_ps(_mm_set1_ps(-16.0f32), y, _mm_set1_ps(116.0f32));
    let a = _mm_mul_ps(_mm_sub_ps(x, y), _mm_set1_ps(500f32));
    let b = _mm_mul_ps(_mm_sub_ps(y, z), _mm_set1_ps(200f32));
    (l, a, b)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
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
