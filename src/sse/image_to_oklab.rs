/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::sse::{
    _mm_color_matrix_ps, get_sse_linear_transfer, sse_deinterleave_rgb, sse_deinterleave_rgba,
    sse_interleave_ps_rgb, sse_interleave_ps_rgba,
};
use crate::{
    load_u8_and_deinterleave, store_and_interleave_v3_f32, store_and_interleave_v4_f32,
    TransferFunction,
};
use erydanos::_mm_cbrt_fast_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! triple_to_oklab {
    ($r: expr, $g: expr, $b: expr, $transfer: expr,
    $c0:expr, $c1:expr, $c2: expr, $c3: expr, $c4:expr, $c5: expr, $c6:expr, $c7: expr, $c8: expr,
        $m0: expr, $m1: expr, $m2: expr, $m3: expr, $m4: expr, $m5: expr, $m6: expr, $m7: expr, $m8: expr
    ) => {{
        let u8_scale = _mm_set1_ps(1f32 / 255f32);
        let r_f = _mm_mul_ps(_mm_cvtepi32_ps($r), u8_scale);
        let g_f = _mm_mul_ps(_mm_cvtepi32_ps($g), u8_scale);
        let b_f = _mm_mul_ps(_mm_cvtepi32_ps($b), u8_scale);
        let r_linear = $transfer(r_f);
        let g_linear = $transfer(g_f);
        let b_linear = $transfer(b_f);

        let (l_l, l_m, l_s) = _mm_color_matrix_ps(
            r_linear, g_linear, b_linear, $c0, $c1, $c2, $c3, $c4, $c5, $c6, $c7, $c8,
        );

        let l_ = _mm_cbrt_fast_ps(l_l);
        let m_ = _mm_cbrt_fast_ps(l_m);
        let s_ = _mm_cbrt_fast_ps(l_s);

        let (l, m, s) =
            _mm_color_matrix_ps(l_, m_, s_, $m0, $m1, $m2, $m3, $m4, $m5, $m6, $m7, $m8);
        (l, m, s)
    }};
}

#[inline(always)]
pub unsafe fn sse_image_to_oklab<const CHANNELS_CONFIGURATION: u8>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer = get_sse_linear_transfer(transfer_function);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        _mm_set1_ps(0.4122214708f32),
        _mm_set1_ps(0.5363325363f32),
        _mm_set1_ps(0.0514459929f32),
        _mm_set1_ps(0.2119034982f32),
        _mm_set1_ps(0.6806995451f32),
        _mm_set1_ps(0.1073969566f32),
        _mm_set1_ps(0.0883024619f32),
        _mm_set1_ps(0.2817188376f32),
        _mm_set1_ps(0.6299787005f32),
    );

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        _mm_set1_ps(0.2104542553f32),
        _mm_set1_ps(0.7936177850f32),
        _mm_set1_ps(-0.0040720468f32),
        _mm_set1_ps(1.9779984951f32),
        _mm_set1_ps(-2.4285922050f32),
        _mm_set1_ps(0.4505937099f32),
        _mm_set1_ps(0.0259040371f32),
        _mm_set1_ps(0.7827717662f32),
        _mm_set1_ps(-0.8086757660f32),
    );

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_u8_and_deinterleave!(src_ptr, image_configuration);

        let r_low = _mm_cvtepu8_epi16(r_chan);
        let g_low = _mm_cvtepu8_epi16(g_chan);
        let b_low = _mm_cvtepu8_epi16(b_chan);

        let r_low_low = _mm_cvtepu16_epi32(r_low);
        let g_low_low = _mm_cvtepu16_epi32(g_low);
        let b_low_low = _mm_cvtepu16_epi32(b_low);

        let (x_low_low, y_low_low, z_low_low) = triple_to_oklab!(
            r_low_low, g_low_low, b_low_low, &transfer, c0, c1, c2, c3, c4, c5, c6, c7, c8, m0, m1,
            m2, m3, m4, m5, m6, m7, m8
        );

        let a_low = _mm_cvtepu8_epi16(a_chan);

        let u8_scale = _mm_set1_ps(1f32 / 255f32);

        if image_configuration.has_alpha() {
            let a_low_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_low)), u8_scale);
            let ptr = dst_ptr.add(cx * 4);
            store_and_interleave_v4_f32!(ptr, x_low_low, y_low_low, z_low_low, a_low_low);
        } else {
            let ptr = dst_ptr.add(cx * 3);
            store_and_interleave_v3_f32!(ptr, x_low_low, y_low_low, z_low_low);
        }

        let r_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(r_low));
        let g_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(g_low));
        let b_low_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b_low));

        let (x_low_high, y_low_high, z_low_high) = triple_to_oklab!(
            r_low_high, g_low_high, b_low_high, &transfer, c0, c1, c2, c3, c4, c5, c6, c7, c8, m0,
            m1, m2, m3, m4, m5, m6, m7, m8
        );

        if image_configuration.has_alpha() {
            let a_low_high = _mm_mul_ps(
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128::<8>(a_low))),
                u8_scale,
            );

            let ptr = dst_ptr.add(cx * 4 + 16);
            store_and_interleave_v4_f32!(ptr, x_low_high, y_low_high, z_low_high, a_low_high);
        } else {
            let ptr = dst_ptr.add(cx * 3 + 4 * 3);
            store_and_interleave_v3_f32!(ptr, x_low_high, y_low_high, z_low_high);
        }

        let r_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(r_chan));
        let g_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(g_chan));
        let b_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(b_chan));

        let r_high_low = _mm_cvtepu16_epi32(r_high);
        let g_high_low = _mm_cvtepu16_epi32(g_high);
        let b_high_low = _mm_cvtepu16_epi32(b_high);

        let (x_high_low, y_high_low, z_high_low) = triple_to_oklab!(
            r_high_low, g_high_low, b_high_low, &transfer, c0, c1, c2, c3, c4, c5, c6, c7, c8, m0,
            m1, m2, m3, m4, m5, m6, m7, m8
        );

        let a_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(a_chan));

        if image_configuration.has_alpha() {
            let a_high_low = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_high)), u8_scale);
            let ptr = dst_ptr.add(cx * 4 + 4 * 4 * 2);
            store_and_interleave_v4_f32!(ptr, x_high_low, y_high_low, z_high_low, a_high_low);
        } else {
            let ptr = dst_ptr.add(cx * 3 + 4 * 3 * 2);
            store_and_interleave_v3_f32!(ptr, x_high_low, y_high_low, z_high_low);
        }

        let r_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(r_high));
        let g_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(g_high));
        let b_high_high = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b_high));

        let (x_high_high, y_high_high, z_high_high) = triple_to_oklab!(
            r_high_high,
            g_high_high,
            b_high_high,
            &transfer,
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
            let a_high_high = _mm_mul_ps(
                _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128::<8>(a_high))),
                u8_scale,
            );
            let ptr = dst_ptr.add(cx * 4 + 4 * 4 * 3);
            store_and_interleave_v4_f32!(ptr, x_high_high, y_high_high, z_high_high, a_high_high);
        } else {
            let ptr = dst_ptr.add(cx * 3 + 4 * 3 * 3);
            store_and_interleave_v3_f32!(ptr, x_high_high, y_high_high, z_high_high);
        }

        cx += 16;
    }

    cx
}
