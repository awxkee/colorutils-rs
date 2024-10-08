/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::sse::{_mm_loadu_ps_x4, _mm_storeu_si128_x4, perform_sse_gamma_transfer};
use crate::TransferFunction;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn transfer_to_gamma(r: __m128, transfer_function: TransferFunction) -> __m128i {
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(
        perform_sse_gamma_transfer(transfer_function, r),
        _mm_set1_ps(255f32),
    )))
}

#[inline(always)]
unsafe fn process_set(
    k: (__m128, __m128, __m128, __m128),
    transfer_function: TransferFunction,
) -> __m128i {
    let y0 = transfer_to_gamma(k.0, transfer_function);
    let y1 = transfer_to_gamma(k.1, transfer_function);
    let y2 = transfer_to_gamma(k.2, transfer_function);
    let y3 = transfer_to_gamma(k.3, transfer_function);

    let y_row01 = _mm_packus_epi32(y0, y1);
    let y_row23 = _mm_packus_epi32(y2, y3);

    _mm_packus_epi16(y_row01, y_row23)
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_linear_plane_to_gamma(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    transfer_function: TransferFunction,
) -> usize {
    let mut cx = start_cx;

    while cx + 64 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset as usize) as *const f32).add(cx);

        let pixel_row0 = _mm_loadu_ps_x4(offset_src_ptr);
        let pixel_row1 = _mm_loadu_ps_x4(offset_src_ptr.add(16));
        let pixel_row2 = _mm_loadu_ps_x4(offset_src_ptr.add(32));
        let pixel_row3 = _mm_loadu_ps_x4(offset_src_ptr.add(48));

        let set0 = process_set(pixel_row0, transfer_function);
        let set1 = process_set(pixel_row1, transfer_function);
        let set2 = process_set(pixel_row2, transfer_function);
        let set3 = process_set(pixel_row3, transfer_function);

        let dst_ptr = dst.add(dst_offset as usize + cx);

        _mm_storeu_si128_x4(dst_ptr, (set0, set1, set2, set3));

        cx += 64;
    }

    while cx + 16 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset as usize) as *const f32).add(cx);

        let pixel_row = _mm_loadu_ps_x4(offset_src_ptr);
        let r_row = process_set(pixel_row, transfer_function);
        let dst_ptr = dst.add(dst_offset as usize + cx);
        _mm_storeu_si128(dst_ptr as *mut __m128i, r_row);

        cx += 16;
    }

    cx
}
