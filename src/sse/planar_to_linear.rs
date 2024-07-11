use crate::sse::{_mm_loadu_si128_x4, _mm_storeu_ps_x4, get_sse_linear_transfer};
use crate::TransferFunction;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_to_linear(r: __m128i, transfer: &unsafe fn(__m128) -> __m128) -> __m128 {
    let r_f = _mm_mul_ps(_mm_cvtepi32_ps(r), _mm_set1_ps(1f32 / 255f32));
    transfer(r_f)
}

#[inline]
unsafe fn process_pixels(
    pixels: __m128i,
    transfer: &unsafe fn(__m128) -> __m128,
) -> (__m128, __m128, __m128, __m128) {
    let zeros = _mm_setzero_si128();
    let r_low = _mm_unpacklo_epi8(pixels, zeros);

    let r_low_low = _mm_unpacklo_epi16(r_low, zeros);

    let x_low_low = sse_to_linear(r_low_low, &transfer);

    let r_low_high = _mm_unpackhi_epi16(r_low, zeros);

    let x_low_high = sse_to_linear(r_low_high, &transfer);

    let r_high = _mm_unpackhi_epi8(pixels, zeros);

    let r_high_low = _mm_unpacklo_epi16(r_high, zeros);

    let x_high_low = sse_to_linear(r_high_low, &transfer);

    let r_high_high = _mm_unpackhi_epi16(r_high, zeros);

    let x_high_high = sse_to_linear(r_high_high, &transfer);

    (x_low_low, x_low_high, x_high_low, x_high_high)
}

#[inline(always)]
pub unsafe fn sse_plane_to_linear(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    transfer_function: TransferFunction,
) -> usize {
    let mut cx = start_cx;
    let transfer = get_sse_linear_transfer(transfer_function);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 64 < width as usize {
        let src_ptr = src.add(src_offset + cx);
        let pixels_row64 = _mm_loadu_si128_x4(src_ptr);
        let storing_row0 = process_pixels(pixels_row64.0, &transfer);
        _mm_storeu_ps_x4(dst_ptr.add(cx), storing_row0);

        let storing_row1 = process_pixels(pixels_row64.1, &transfer);
        _mm_storeu_ps_x4(dst_ptr.add(cx + 16), storing_row1);

        let storing_row2 = process_pixels(pixels_row64.2, &transfer);
        _mm_storeu_ps_x4(dst_ptr.add(cx + 32), storing_row2);

        let storing_row3 = process_pixels(pixels_row64.3, &transfer);
        _mm_storeu_ps_x4(dst_ptr.add(cx + 48), storing_row3);

        cx += 64;
    }

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx);
        let pixels = _mm_loadu_si128(src_ptr as *const __m128i);
        let storing_row = process_pixels(pixels, &transfer);
        _mm_storeu_ps_x4(dst_ptr.add(cx), storing_row);

        cx += 16;
    }

    cx
}
