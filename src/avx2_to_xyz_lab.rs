#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx2_utils::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx_gamma_curves::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx_math::*;
#[allow(unused_imports)]
use crate::gamma_curves::TransferFunction;
#[allow(unused_imports)]
use crate::image::ImageConfiguration;
#[allow(unused_imports)]
use crate::image_to_xyz_lab::XyzTarget;
#[allow(unused_imports)]
use crate::neon_gamma_curves::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(unused_imports)]
use crate::sse_gamma_curves::{sse_rec709_to_linear, sse_srgb_to_linear};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::x86_64_simd_support::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn get_avx2_linear_transfer(
    transfer_function: TransferFunction,
) -> unsafe fn(__m256) -> __m256 {
    match transfer_function {
        TransferFunction::Srgb => avx2_srgb_to_linear,
        TransferFunction::Rec709 => avx2_rec709_to_linear,
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx2_triple_to_xyz(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    c1: __m256,
    c2: __m256,
    c3: __m256,
    c4: __m256,
    c5: __m256,
    c6: __m256,
    c7: __m256,
    c8: __m256,
    c9: __m256,
    transfer: &unsafe fn(__m256) -> __m256,
) -> (__m256, __m256, __m256) {
    let u8_scale = _mm256_set1_ps(1f32 / 255f32);
    let r_f = _mm256_mul_ps(_mm256_cvtepi32_ps(r), u8_scale);
    let g_f = _mm256_mul_ps(_mm256_cvtepi32_ps(g), u8_scale);
    let b_f = _mm256_mul_ps(_mm256_cvtepi32_ps(b), u8_scale);
    let r_linear = transfer(r_f);
    let g_linear = transfer(g_f);
    let b_linear = transfer(b_f);

    let (x, y, z) = _mm256_color_matrix_ps(
        r_linear, g_linear, b_linear, c1, c2, c3, c4, c5, c6, c7, c8, c9,
    );
    (x, y, z)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx2_triple_to_lab(x: __m256, y: __m256, z: __m256) -> (__m256, __m256, __m256) {
    let x = _mm256_mul_ps(x, _mm256_set1_ps(100f32 / 95.047f32));
    let y = _mm256_mul_ps(y, _mm256_set1_ps(100f32 / 100f32));
    let z = _mm256_mul_ps(z, _mm256_set1_ps(100f32 / 108.883f32));
    let cbrt_x = _mm256_cbrt_ps(x);
    let cbrt_y = _mm256_cbrt_ps(y);
    let cbrt_z = _mm256_cbrt_ps(z);
    let s_1 = _mm256_set1_ps(16.0 / 116.0);
    let s_2 = _mm256_set1_ps(7.787);
    let lower_x = _mm256_prefer_fma_ps(s_1, s_2, x);
    let lower_y = _mm256_prefer_fma_ps(s_1, s_2, y);
    let lower_z = _mm256_prefer_fma_ps(s_1, s_2, z);
    let cutoff = _mm256_set1_ps(0.008856f32);
    let x = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(x, cutoff), cbrt_x, lower_x);
    let y = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(y, cutoff), cbrt_y, lower_y);
    let z = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(z, cutoff), cbrt_z, lower_z);
    let l = _mm256_prefer_fma_ps(_mm256_set1_ps(-16.0f32), y, _mm256_set1_ps(116.0f32));
    let a = _mm256_mul_ps(_mm256_sub_ps(x, y), _mm256_set1_ps(500f32));
    let b = _mm256_mul_ps(_mm256_sub_ps(y, z), _mm256_set1_ps(200f32));
    (l, a, b)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn avx2_channels_to_xyz_or_lab<
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

    let transfer = get_avx2_linear_transfer(transfer_function);

    let cq1 = _mm256_set1_ps(matrix[0][0]);
    let cq2 = _mm256_set1_ps(matrix[0][1]);
    let cq3 = _mm256_set1_ps(matrix[0][2]);
    let cq4 = _mm256_set1_ps(matrix[1][0]);
    let cq5 = _mm256_set1_ps(matrix[1][1]);
    let cq6 = _mm256_set1_ps(matrix[1][2]);
    let cq7 = _mm256_set1_ps(matrix[2][0]);
    let cq8 = _mm256_set1_ps(matrix[2][1]);
    let cq9 = _mm256_set1_ps(matrix[2][2]);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 32 < width as usize {
        let (r_chan, g_chan, b_chan, a_chan);
        let src_ptr = src.add(src_offset + cx * channels);
        let row1 = _mm256_loadu_si256(src_ptr as *const __m256i);
        let row2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
        let row3 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let (c1, c2, c3) = avx2_deinterleave_rgb(row1, row2, row3);
                if image_configuration == ImageConfiguration::Rgb {
                    r_chan = c1;
                    g_chan = c2;
                    b_chan = c3;
                } else {
                    r_chan = c3;
                    g_chan = c2;
                    b_chan = c1;
                }
                a_chan = _mm256_set1_epi8(0);
            }
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
                let row4 = _mm256_loadu_si256(src_ptr.add(64 + 32) as *const __m256i);
                let (c1, c2, c3, c4) = avx2_deinterleave_rgba(row1, row2, row3, row4);
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

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (mut x_low_low, mut y_low_low, mut z_low_low) = avx2_triple_to_xyz(
            r_low_low, g_low_low, b_low_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9, &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = avx2_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::XYZ => {}
        }

        let write_dst_ptr = dst_ptr.add(cx * 3);

        let (v0, v1, v2) = avx2_interleave_rgb_ps(x_low_low, y_low_low, z_low_low);

        _mm256_storeu_ps(write_dst_ptr, v0);
        _mm256_storeu_ps(write_dst_ptr.add(8), v1);
        _mm256_storeu_ps(write_dst_ptr.add(16), v2);

        let r_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r_low));
        let g_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(g_low));
        let b_low_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(b_low));

        let (mut x_low_high, mut y_low_high, mut z_low_high) = avx2_triple_to_xyz(
            r_low_high, g_low_high, b_low_high, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = avx2_triple_to_lab(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = a;
                z_low_high = b;
            }
            XyzTarget::XYZ => {}
        }

        let (v0, v1, v2) = avx2_interleave_rgb_ps(x_low_high, y_low_high, z_low_high);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3), v0);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 + 8), v1);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 + 16), v2);

        let r_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_chan));
        let g_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_chan));
        let b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_chan));

        let r_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_high));
        let g_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_high));
        let b_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_high));

        let (mut x_high_low, mut y_high_low, mut z_high_low) = avx2_triple_to_xyz(
            r_high_low, g_high_low, b_high_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = avx2_triple_to_lab(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = a;
                z_high_low = b;
            }
            XyzTarget::XYZ => {}
        }

        let (v0, v1, v2) = avx2_interleave_rgb_ps(x_high_low, y_high_low, z_high_low);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 2), v0);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 2 + 8), v1);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 2 + 16), v2);

        let r_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(r_high));
        let g_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(g_high));
        let b_high_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(b_high));

        let (mut x_high_high, mut y_high_high, mut z_high_high) = avx2_triple_to_xyz(
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
                let (l, a, b) = avx2_triple_to_lab(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = a;
                z_high_high = b;
            }
            XyzTarget::XYZ => {}
        }

        let (v0, v1, v2) = avx2_interleave_rgb_ps(x_high_high, y_high_high, z_high_high);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 3), v0);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 3 + 8), v1);
        _mm256_storeu_ps(write_dst_ptr.add(8 * 3 * 3 + 16), v2);

        if USE_ALPHA {
            let a_ptr = (a_linearized as *mut u8).add(a_offset) as *mut f32;

            let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

            let u8_scale = _mm256_set1_ps(1f32 / 255f32);

            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx), a_low_low);

            let a_low_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a_low))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx + 8), a_low_high);

            let a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a_chan));

            let a_high_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_high))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx + 8 * 2), a_high_low);

            let a_high_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a_high))),
                u8_scale,
            );

            _mm256_storeu_ps(a_ptr.add(cx + 8 * 3), a_high_high);
        }

        cx += 32;
    }

    cx
}
