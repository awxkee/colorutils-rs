/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::neon::get_neon_gamma_transfer;
use crate::TransferFunction;

#[inline(always)]
unsafe fn transfer_to_gamma(
    r: float32x4_t,
    transfer: &unsafe fn(float32x4_t) -> float32x4_t,
) -> uint32x4_t {
    let r_f = vcvtaq_u32_f32(vmulq_n_f32(transfer(r), 255f32));
    r_f
}

#[inline(always)]
unsafe fn process_set(
    k: float32x4x4_t,
    function: &unsafe fn(float32x4_t) -> float32x4_t,
) -> uint8x16_t {
    let y0 = transfer_to_gamma(k.0, &function);
    let y1 = transfer_to_gamma(k.1, &function);
    let y2 = transfer_to_gamma(k.2, &function);
    let y3 = transfer_to_gamma(k.3, &function);

    let y_row01 = vcombine_u16(vqmovn_u32(y0), vqmovn_u32(y1));
    let y_row23 = vcombine_u16(vqmovn_u32(y2), vqmovn_u32(y3));

    let r_row = vcombine_u8(vqmovn_u16(y_row01), vqmovn_u16(y_row23));
    r_row
}

#[inline]
pub unsafe fn neon_linear_plane_to_gamma(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    transfer_function: TransferFunction,
) -> usize {
    let mut cx = start_cx;

    let function = get_neon_gamma_transfer(transfer_function);

    while cx + 64 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset as usize) as *const f32).add(cx);

        let pixel_row0 = vld1q_f32_x4(offset_src_ptr);
        let pixel_row1 = vld1q_f32_x4(offset_src_ptr.add(16));
        let pixel_row2 = vld1q_f32_x4(offset_src_ptr.add(32));
        let pixel_row3 = vld1q_f32_x4(offset_src_ptr.add(48));

        let set0 = process_set(pixel_row0, &function);
        let set1 = process_set(pixel_row1, &function);
        let set2 = process_set(pixel_row2, &function);
        let set3 = process_set(pixel_row3, &function);

        let dst_ptr = dst.add(dst_offset as usize + cx);

        let pixel_set = uint8x16x4_t(set0, set1, set2, set3);
        vst1q_u8_x4(dst_ptr, pixel_set);

        cx += 64;
    }

    while cx + 16 < width as usize {
        let offset_src_ptr = ((src as *const u8).add(src_offset as usize) as *const f32).add(cx);

        let pixel_row = vld1q_f32_x4(offset_src_ptr);
        let r_row = process_set(pixel_row, &function);
        let dst_ptr = dst.add(dst_offset as usize + cx);
        vst1q_u8(dst_ptr, r_row);

        cx += 16;
    }

    cx
}
