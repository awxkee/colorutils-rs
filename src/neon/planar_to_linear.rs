/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::neon::*;
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_to_linear(r: uint32x4_t, transfer_function: TransferFunction) -> float32x4_t {
    let r_f = vmulq_n_f32(vcvtq_f32_u32(r), 1f32 / 255f32);
    neon_perform_linear_transfer(transfer_function, r_f)
}

#[inline]
unsafe fn process_pixels(pixels: uint8x16_t, transfer_function: TransferFunction) -> float32x4x4_t {
    let r_low = vmovl_u8(vget_low_u8(pixels));

    let r_low_low = vmovl_u16(vget_low_u16(r_low));

    let x_low_low = neon_to_linear(r_low_low, transfer_function);

    let r_low_high = vmovl_high_u16(r_low);

    let x_low_high = neon_to_linear(r_low_high, transfer_function);

    let r_high = vmovl_high_u8(pixels);

    let r_high_low = vmovl_u16(vget_low_u16(r_high));

    let x_high_low = neon_to_linear(r_high_low, transfer_function);

    let r_high_high = vmovl_high_u16(r_high);

    let x_high_high = neon_to_linear(r_high_high, transfer_function);
    float32x4x4_t(x_low_low, x_low_high, x_high_low, x_high_high)
}

#[inline(always)]
pub unsafe fn neon_plane_to_linear(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    transfer_function: TransferFunction,
) -> usize {
    let mut cx = start_cx;

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 64 < width as usize {
        let src_ptr = src.add(src_offset + cx);
        let pixels_row64 = vld1q_u8_x4(src_ptr);
        let storing_row0 = process_pixels(pixels_row64.0, transfer_function);
        vst1q_f32_x4(dst_ptr.add(cx), storing_row0);

        let storing_row1 = process_pixels(pixels_row64.1, transfer_function);
        vst1q_f32_x4(dst_ptr.add(cx + 16), storing_row1);

        let storing_row2 = process_pixels(pixels_row64.2, transfer_function);
        vst1q_f32_x4(dst_ptr.add(cx + 32), storing_row2);

        let storing_row3 = process_pixels(pixels_row64.3, transfer_function);
        vst1q_f32_x4(dst_ptr.add(cx + 48), storing_row3);

        cx += 64;
    }

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx);
        let pixels = vld1q_u8(src_ptr);
        let storing_row = process_pixels(pixels, transfer_function);
        vst1q_f32_x4(dst_ptr.add(cx), storing_row);

        cx += 16;
    }

    cx
}
