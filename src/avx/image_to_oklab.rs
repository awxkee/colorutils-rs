/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use erydanos::{_mm256_atan2_ps, _mm256_cbrt_fast_ps, _mm256_hypot_fast_ps};

use crate::avx::gamma_curves::get_avx2_linear_transfer;
use crate::avx::routines::{
    avx_vld_u8_and_deinterleave, avx_vld_u8_and_deinterleave_half,
    avx_vld_u8_and_deinterleave_quarter,
};
use crate::avx::{_mm256_color_matrix_ps, avx2_interleave_rgb_ps, avx2_interleave_rgba_ps};
use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
use crate::{
    avx_store_and_interleave_v3_direct_f32, avx_store_and_interleave_v4_direct_f32,
    TransferFunction, SRGB_TO_XYZ_D65,
};

macro_rules! triple_to_oklab {
    ($r: expr, $g: expr, $b: expr, $transfer: expr, $target: expr,
        $x0: expr, $x1: expr, $x2: expr, $x3: expr, $x4: expr, $x5: expr, $x6: expr, $x7: expr, $x8: expr,
    $c0:expr, $c1:expr, $c2: expr, $c3: expr, $c4:expr, $c5: expr, $c6:expr, $c7: expr, $c8: expr,
        $m0: expr, $m1: expr, $m2: expr, $m3: expr, $m4: expr, $m5: expr, $m6: expr, $m7: expr, $m8: expr
    ) => {{
        let u8_scale = _mm256_set1_ps(1f32 / 255f32);
        let r_f = _mm256_mul_ps(_mm256_cvtepi32_ps($r), u8_scale);
        let g_f = _mm256_mul_ps(_mm256_cvtepi32_ps($g), u8_scale);
        let b_f = _mm256_mul_ps(_mm256_cvtepi32_ps($b), u8_scale);
        let r_linear = $transfer(r_f);
        let g_linear = $transfer(g_f);
        let b_linear = $transfer(b_f);

        let (x, y, z) = _mm256_color_matrix_ps(
            r_linear, g_linear, b_linear, $x0, $x1, $x2, $x3, $x4, $x5, $x6, $x7, $x8,
        );

        let (l_l, l_m, l_s) =
            _mm256_color_matrix_ps(x, y, z, $c0, $c1, $c2, $c3, $c4, $c5, $c6, $c7, $c8);

        let l_ = _mm256_cbrt_fast_ps(l_l);
        let m_ = _mm256_cbrt_fast_ps(l_m);
        let s_ = _mm256_cbrt_fast_ps(l_s);

        let (l, mut a, mut b) =
            _mm256_color_matrix_ps(l_, m_, s_, $m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, $m8);

        if $target == OklabTarget::OKLCH {
            let c = _mm256_hypot_fast_ps(a, b);
            let h = _mm256_atan2_ps(b, a);
            a = c;
            b = h;
        }

        (l, a, b)
    }};
}

#[inline(always)]
pub unsafe fn avx_image_to_oklab<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    transfer_function: TransferFunction,
) -> usize {
    let target: OklabTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer = get_avx2_linear_transfer(transfer_function);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    // Matrix To XYZ
    let (x0, x1, x2, x3, x4, x5, x6, x7, x8) = (
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(0)),
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(1)),
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(0).get_unchecked(2)),
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(0)),
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(1)),
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(1).get_unchecked(2)),
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(0)),
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(1)),
        _mm256_set1_ps(*SRGB_TO_XYZ_D65.get_unchecked(2).get_unchecked(2)),
    );

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        _mm256_set1_ps(0.4122214708f32),
        _mm256_set1_ps(0.5363325363f32),
        _mm256_set1_ps(0.0514459929f32),
        _mm256_set1_ps(0.2119034982f32),
        _mm256_set1_ps(0.6806995451f32),
        _mm256_set1_ps(0.1073969566f32),
        _mm256_set1_ps(0.0883024619f32),
        _mm256_set1_ps(0.2817188376f32),
        _mm256_set1_ps(0.6299787005f32),
    );

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        _mm256_set1_ps(0.2104542553f32),
        _mm256_set1_ps(0.7936177850f32),
        _mm256_set1_ps(-0.0040720468f32),
        _mm256_set1_ps(1.9779984951f32),
        _mm256_set1_ps(-2.4285922050f32),
        _mm256_set1_ps(0.4505937099f32),
        _mm256_set1_ps(0.0259040371f32),
        _mm256_set1_ps(0.7827717662f32),
        _mm256_set1_ps(-0.8086757660f32),
    );

    let zeros = _mm256_setzero_si256();

    while cx + 32 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (x_low_low, y_low_low, z_low_low) = triple_to_oklab!(
            r_low_low, g_low_low, b_low_low, &transfer, target, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            c0, c1, c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

        let u8_scale = _mm256_set1_ps(1f32 / 255f32);

        if image_configuration.has_alpha() {
            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );
            let ptr = dst_ptr.add(cx * 4);
            avx_store_and_interleave_v4_direct_f32!(
                ptr, x_low_low, y_low_low, z_low_low, a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3);
            avx_store_and_interleave_v3_direct_f32!(ptr, x_low_low, y_low_low, z_low_low);
        }

        let r_low_high = _mm256_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm256_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm256_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) = triple_to_oklab!(
            r_low_high, g_low_high, b_low_high, &transfer, target, x0, x1, x2, x3, x4, x5, x6, x7,
            x8, c0, c1, c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        if image_configuration.has_alpha() {
            let a_low_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(a_low))),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4 + 32);
            avx_store_and_interleave_v4_direct_f32!(
                ptr, x_low_high, y_low_high, z_low_high, a_low_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 8 * 3);
            avx_store_and_interleave_v3_direct_f32!(ptr, x_low_high, y_low_high, z_low_high);
        }

        let r_high = _mm256_unpackhi_epi8(r_chan, zeros);
        let g_high = _mm256_unpackhi_epi8(g_chan, zeros);
        let b_high = _mm256_unpackhi_epi8(b_chan, zeros);

        let r_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_high));
        let g_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_high));
        let b_high_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_high));

        let (x_high_low, y_high_low, z_high_low) = triple_to_oklab!(
            r_high_low, g_high_low, b_high_low, &transfer, target, x0, x1, x2, x3, x4, x5, x6, x7,
            x8, c0, c1, c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        let a_high = _mm256_unpackhi_epi8(a_chan, zeros);

        if image_configuration.has_alpha() {
            let a_high_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_high))),
                u8_scale,
            );
            let ptr = dst_ptr.add(cx * 4 + 8 * 4 * 2);
            avx_store_and_interleave_v4_direct_f32!(
                ptr, x_high_low, y_high_low, z_high_low, a_high_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 8 * 3 * 2);
            avx_store_and_interleave_v3_direct_f32!(ptr, x_high_low, y_high_low, z_high_low);
        }

        let r_high_high = _mm256_unpackhi_epi16(r_high, zeros);
        let g_high_high = _mm256_unpackhi_epi16(g_high, zeros);
        let b_high_high = _mm256_unpackhi_epi16(b_high, zeros);

        let (x_high_high, y_high_high, z_high_high) = triple_to_oklab!(
            r_high_high,
            g_high_high,
            b_high_high,
            &transfer,
            target,
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8
        );

        if image_configuration.has_alpha() {
            let a_high_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_high, zeros)),
                u8_scale,
            );
            let ptr = dst_ptr.add(cx * 4 + 8 * 4 * 3);
            avx_store_and_interleave_v4_direct_f32!(
                ptr,
                x_high_high,
                y_high_high,
                z_high_high,
                a_high_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 8 * 3 * 3);
            avx_store_and_interleave_v3_direct_f32!(ptr, x_high_high, y_high_high, z_high_high);
        }

        cx += 32;
    }

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave_half::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (x_low_low, y_low_low, z_low_low) = triple_to_oklab!(
            r_low_low, g_low_low, b_low_low, &transfer, target, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            c0, c1, c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

        let u8_scale = _mm256_set1_ps(1f32 / 255f32);

        if image_configuration.has_alpha() {
            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );
            let ptr = dst_ptr.add(cx * 4);
            avx_store_and_interleave_v4_direct_f32!(
                ptr, x_low_low, y_low_low, z_low_low, a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3);
            avx_store_and_interleave_v3_direct_f32!(ptr, x_low_low, y_low_low, z_low_low);
        }

        let r_low_high = _mm256_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm256_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm256_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) = triple_to_oklab!(
            r_low_high, g_low_high, b_low_high, &transfer, target, x0, x1, x2, x3, x4, x5, x6, x7,
            x8, c0, c1, c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        if image_configuration.has_alpha() {
            let a_low_high = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_low, zeros)),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4 + 32);
            avx_store_and_interleave_v4_direct_f32!(
                ptr, x_low_high, y_low_high, z_low_high, a_low_high
            );
        } else {
            let ptr = dst_ptr.add(cx * 3 + 8 * 3);
            avx_store_and_interleave_v3_direct_f32!(ptr, x_low_high, y_low_high, z_low_high);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            avx_vld_u8_and_deinterleave_quarter::<CHANNELS_CONFIGURATION>(src_ptr);

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_chan));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_chan));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_chan));

        let r_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r_low));
        let g_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g_low));
        let b_low_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b_low));

        let (x_low_low, y_low_low, z_low_low) = triple_to_oklab!(
            r_low_low, g_low_low, b_low_low, &transfer, target, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            c0, c1, c2, c3, c4, c5, c6, c7, c8, m0, m1, m2, m3, m4, m5, m6, m7, m8
        );

        let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_chan));

        let u8_scale = _mm256_set1_ps(1f32 / 255f32);

        if image_configuration.has_alpha() {
            let a_low_low = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a_low))),
                u8_scale,
            );
            let ptr = dst_ptr.add(cx * 4);
            avx_store_and_interleave_v4_direct_f32!(
                ptr, x_low_low, y_low_low, z_low_low, a_low_low
            );
        } else {
            let ptr = dst_ptr.add(cx * 3);
            avx_store_and_interleave_v3_direct_f32!(ptr, x_low_low, y_low_low, z_low_low);
        }

        cx += 8;
    }

    cx
}
