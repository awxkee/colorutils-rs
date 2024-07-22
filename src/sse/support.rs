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

pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub unsafe fn sse_interleave_rgba(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let rg_lo = _mm_unpacklo_epi8(r, g);
    let rg_hi = _mm_unpackhi_epi8(r, g);
    let ba_lo = _mm_unpacklo_epi8(b, a);
    let ba_hi = _mm_unpackhi_epi8(b, a);

    let rgba_0_lo = _mm_unpacklo_epi16(rg_lo, ba_lo);
    let rgba_0_hi = _mm_unpackhi_epi16(rg_lo, ba_lo);
    let rgba_1_lo = _mm_unpacklo_epi16(rg_hi, ba_hi);
    let rgba_1_hi = _mm_unpackhi_epi16(rg_hi, ba_hi);
    (rgba_0_lo, rgba_0_hi, rgba_1_lo, rgba_1_hi)
}

#[inline(always)]
pub unsafe fn sse_interleave_ps_rgb(a: __m128, b: __m128, c: __m128) -> (__m128, __m128, __m128) {
    const MASK_U0: i32 = shuffle(0, 0, 0, 0);
    let u0 = _mm_shuffle_ps::<MASK_U0>(a, b);
    const MASK_U1: i32 = shuffle(1, 1, 0, 0);
    let u1 = _mm_shuffle_ps::<MASK_U1>(c, a);
    const MASK_V0: i32 = shuffle(2, 0, 2, 0);
    let v0 = _mm_shuffle_ps::<MASK_V0>(u0, u1);
    const MASK_U2: i32 = shuffle(1, 1, 1, 1);
    let u2 = _mm_shuffle_ps::<MASK_U2>(b, c);
    const MASK_U3: i32 = shuffle(2, 2, 2, 2);
    let u3 = _mm_shuffle_ps::<MASK_U3>(a, b);
    const MASK_V1: i32 = shuffle(2, 0, 2, 0);
    let v1 = _mm_shuffle_ps::<MASK_V1>(u2, u3);
    const MASK_U4: i32 = shuffle(3, 3, 2, 2);
    let u4 = _mm_shuffle_ps::<MASK_U4>(c, a);
    const MASK_U5: i32 = shuffle(3, 3, 3, 3);
    let u5 = _mm_shuffle_ps::<MASK_U5>(b, c);
    const MASK_V2: i32 = shuffle(2, 0, 2, 0);
    let v2 = _mm_shuffle_ps::<MASK_V2>(u4, u5);
    (v0, v1, v2)
}

#[inline(always)]
pub unsafe fn sse_interleave_ps_rgba(
    a: __m128,
    b: __m128,
    c: __m128,
    d: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let u0 = _mm_unpacklo_ps(a, c);
    let u1 = _mm_unpacklo_ps(b, d);
    let u2 = _mm_unpackhi_ps(a, c);
    let u3 = _mm_unpackhi_ps(b, d);
    let v0 = _mm_unpacklo_ps(u0, u1);
    let v2 = _mm_unpacklo_ps(u2, u3);
    let v1 = _mm_unpackhi_ps(u0, u1);
    let v3 = _mm_unpackhi_ps(u2, u3);
    (v0, v1, v2, v3)
}

#[inline(always)]
pub unsafe fn sse_deinterleave_rgba(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
    rgba3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi8(rgba0, rgba1); // r1 R1 g1 G1 b1 B1 a1 A1 r2 R2 g2 G2 b2 B2 a2 A2
    let t1 = _mm_unpackhi_epi8(rgba0, rgba1);
    let t2 = _mm_unpacklo_epi8(rgba2, rgba3); // r4 R4 g4 G4 b4 B4 a4 A4 r5 R5 g5 G5 b5 B5 a5 A5
    let t3 = _mm_unpackhi_epi8(rgba2, rgba3);

    let t4 = _mm_unpacklo_epi16(t0, t2); // r1 R1 r4 R4 g1 G1 G4 g4 G4 b1 B1 b4 B4 a1 A1 a4 A4
    let t5 = _mm_unpackhi_epi16(t0, t2);
    let t6 = _mm_unpacklo_epi16(t1, t3);
    let t7 = _mm_unpackhi_epi16(t1, t3);

    let l1 = _mm_unpacklo_epi32(t4, t6); // r1 R1 r4 R4 g1 G1 G4 g4 G4 b1 B1 b4 B4 a1 A1 a4 A4
    let l2 = _mm_unpackhi_epi32(t4, t6);
    let l3 = _mm_unpacklo_epi32(t5, t7);
    let l4 = _mm_unpackhi_epi32(t5, t7);

    #[rustfmt::skip]
        let shuffle = _mm_setr_epi8(0, 4, 8, 12,
                                    1, 5, 9, 13,
                                    2, 6, 10, 14,
                                    3, 7, 11, 15,
    );

    let r1 = _mm_shuffle_epi8(_mm_unpacklo_epi32(l1, l3), shuffle);
    let r2 = _mm_shuffle_epi8(_mm_unpackhi_epi32(l1, l3), shuffle);
    let r3 = _mm_shuffle_epi8(_mm_unpacklo_epi32(l2, l4), shuffle);
    let r4 = _mm_shuffle_epi8(_mm_unpackhi_epi32(l2, l4), shuffle);

    (r1, r2, r3, r4)
}

#[inline(always)]
pub unsafe fn sse_deinterleave_rgb_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
) -> (__m128, __m128, __m128) {
    const MASK_AT12: i32 = shuffle(0, 1, 0, 2);
    let at12 = _mm_shuffle_ps::<MASK_AT12>(t1, t2);
    const MASK_V0: i32 = shuffle(2, 0, 3, 0);
    let v0 = _mm_shuffle_ps::<MASK_V0>(t0, at12);
    const MASK_BT01: i32 = shuffle(0, 0, 0, 1);
    let bt01 = _mm_shuffle_ps::<MASK_BT01>(t0, t1);
    const MASK_BT12: i32 = shuffle(0, 2, 0, 3);
    let bt12 = _mm_shuffle_ps::<MASK_BT12>(t1, t2);
    const MASK_V1: i32 = shuffle(2, 0, 2, 0);
    let v1 = _mm_shuffle_ps::<MASK_V1>(bt01, bt12);
    const MASK_CT01: i32 = shuffle(0, 1, 0, 2);
    let ct01 = _mm_shuffle_ps::<MASK_CT01>(t0, t1);
    const MASK_V2: i32 = shuffle(3, 0, 2, 0);
    let v2 = _mm_shuffle_ps::<MASK_V2>(ct01, t2);
    (v0, v1, v2)
}

#[inline(always)]
pub unsafe fn sse_deinterleave_rgb(
    rgb0: __m128i,
    rgb1: __m128i,
    rgb2: __m128i,
) -> (__m128i, __m128i, __m128i) {
    #[rustfmt::skip]
        let idx = _mm_setr_epi8(0, 3, 6, 9,
                                12, 15, 2, 5, 8,
                                11, 14, 1, 4, 7,
                                10, 13);

    let r6b5g5_0 = _mm_shuffle_epi8(rgb0, idx);
    let g6r5b5_1 = _mm_shuffle_epi8(rgb1, idx);
    let b6g5r5_2 = _mm_shuffle_epi8(rgb2, idx);

    #[rustfmt::skip]
        let mask010 = _mm_setr_epi8(0, 0, 0, 0,
                                    0, 0, -1, -1, -1,
                                    -1, -1, 0, 0, 0,
                                    0, 0);

    #[rustfmt::skip]
        let mask001 = _mm_setr_epi8(0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    -1, -1, -1, -1, -1);

    let b2g2b1 = _mm_blendv_epi8(b6g5r5_2, g6r5b5_1, mask001);
    let b2b0b1 = _mm_blendv_epi8(b2g2b1, r6b5g5_0, mask010);

    let r0r1b1 = _mm_blendv_epi8(r6b5g5_0, g6r5b5_1, mask010);
    let r0r1r2 = _mm_blendv_epi8(r0r1b1, b6g5r5_2, mask001);

    let g1r1g0 = _mm_blendv_epi8(g6r5b5_1, r6b5g5_0, mask001);
    let g1g2g0 = _mm_blendv_epi8(g1r1g0, b6g5r5_2, mask010);

    let g0g1g2 = _mm_alignr_epi8::<11>(g1g2g0, g1g2g0);
    let b0b1b2 = _mm_alignr_epi8::<6>(b2b0b1, b2b0b1);

    (r0r1r2, g0g1g2, b0b1b2)
}

#[inline(always)]
pub unsafe fn sse_interleave_rgb(
    r: __m128i,
    g: __m128i,
    b: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    let sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    let sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
    let a0 = _mm_shuffle_epi8(r, sh_a);
    let b0 = _mm_shuffle_epi8(g, sh_b);
    let c0 = _mm_shuffle_epi8(b, sh_c);

    let m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    let m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    let v0 = _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0);
    let v1 = _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0);
    let v2 = _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0);
    (v0, v1, v2)
}

#[inline(always)]
pub unsafe fn sse_interleave_rgb_epi16(
    a: __m128i,
    b: __m128i,
    c: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    let sh_b = _mm_setr_epi8(10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);
    let sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    let a0 = _mm_shuffle_epi8(a, sh_a);
    let b0 = _mm_shuffle_epi8(b, sh_b);
    let c0 = _mm_shuffle_epi8(c, sh_c);

    let v0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(a0, b0), c0);
    let v1 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(c0, a0), b0);
    let v2 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(b0, c0), a0);
    (v0, v1, v2)
}

#[inline(always)]
pub unsafe fn sse_interleave_rgba_epi16(
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let u0 = _mm_unpacklo_epi16(a, c); // a0 c0 a1 c1 ...
    let u1 = _mm_unpackhi_epi16(a, c); // a4 c4 a5 c5 ...
    let u2 = _mm_unpacklo_epi16(b, d); // b0 d0 b1 d1 ...
    let u3 = _mm_unpackhi_epi16(b, d); // b4 d4 b5 d5 ...

    let v0 = _mm_unpacklo_epi16(u0, u2); // a0 b0 c0 d0 ...
    let v1 = _mm_unpackhi_epi16(u0, u2); // a2 b2 c2 d2 ...
    let v2 = _mm_unpacklo_epi16(u1, u3); // a4 b4 c4 d4 ...
    let v3 = _mm_unpackhi_epi16(u1, u3); // a6 b6 c6 d6 ...
    (v0, v1, v2, v3)
}

#[inline(always)]
pub unsafe fn sse_deinterleave_rgba_epi16(
    u0: __m128i,
    u1: __m128i,
    u2: __m128i,
    u3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let v0 = _mm_unpacklo_epi16(u0, u2); // a0 a4 b0 b4 ...
    let v1 = _mm_unpackhi_epi16(u0, u2); // a1 a5 b1 b5 ...
    let v2 = _mm_unpacklo_epi16(u1, u3); // a2 a6 b2 b6 ...
    let v3 = _mm_unpackhi_epi16(u1, u3); // a3 a7 b3 b7 ...

    let u0 = _mm_unpacklo_epi16(v0, v2); // a0 a2 a4 a6 ...
    let u1 = _mm_unpacklo_epi16(v1, v3); // a1 a3 a5 a7 ...
    let u2 = _mm_unpackhi_epi16(v0, v2); // c0 c2 c4 c6 ...
    let u3 = _mm_unpackhi_epi16(v1, v3); // c1 c3 c5 c7 ...

    let a = _mm_unpacklo_epi16(u0, u1);
    let b = _mm_unpackhi_epi16(u0, u1);
    let c = _mm_unpacklo_epi16(u2, u3);
    let d = _mm_unpackhi_epi16(u2, u3);
    (a, b, c, d)
}

#[inline(always)]
pub unsafe fn sse_deinterleave_rgb_epi16(
    v0: __m128i,
    v1: __m128i,
    v2: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let a0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(v0, v1), v2);
    let b0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(v2, v0), v1);
    let c0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(v1, v2), v0);

    let sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    let sh_b = _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
    let sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    let a0 = _mm_shuffle_epi8(a0, sh_a);
    let b0 = _mm_shuffle_epi8(b0, sh_b);
    let c0 = _mm_shuffle_epi8(c0, sh_c);
    (a0, b0, c0)
}

#[inline(always)]
pub unsafe fn sse_deinterleave_rgba_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
    t3: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let t02lo = _mm_unpacklo_ps(t0, t2);
    let t13lo = _mm_unpacklo_ps(t1, t3);
    let t02hi = _mm_unpackhi_ps(t0, t2);
    let t13hi = _mm_unpackhi_ps(t1, t3);
    let v0 = _mm_unpacklo_ps(t02lo, t13lo);
    let v1 = _mm_unpackhi_ps(t02lo, t13lo);
    let v2 = _mm_unpacklo_ps(t02hi, t13hi);
    let v3 = _mm_unpackhi_ps(t02hi, t13hi);
    (v0, v1, v2, v3)
}

#[inline(always)]
pub unsafe fn _mm_loadu_si128_x4(ptr: *const u8) -> (__m128i, __m128i, __m128i, __m128i) {
    (
        _mm_loadu_si128(ptr as *const __m128i),
        _mm_loadu_si128(ptr.add(16) as *const __m128i),
        _mm_loadu_si128(ptr.add(32) as *const __m128i),
        _mm_loadu_si128(ptr.add(48) as *const __m128i),
    )
}

#[inline(always)]
pub unsafe fn _mm_storeu_ps_x4(ptr: *mut f32, set: (__m128, __m128, __m128, __m128)) {
    _mm_storeu_ps(ptr, set.0);
    _mm_storeu_ps(ptr.add(4), set.1);
    _mm_storeu_ps(ptr.add(8), set.2);
    _mm_storeu_ps(ptr.add(12), set.3);
}

#[inline(always)]
pub unsafe fn _mm_loadu_ps_x4(ptr: *const f32) -> (__m128, __m128, __m128, __m128) {
    (
        _mm_loadu_ps(ptr),
        _mm_loadu_ps(ptr.add(4)),
        _mm_loadu_ps(ptr.add(8)),
        _mm_loadu_ps(ptr.add(12)),
    )
}

#[inline(always)]
pub unsafe fn _mm_storeu_si128_x4(ptr: *mut u8, set: (__m128i, __m128i, __m128i, __m128i)) {
    _mm_storeu_si128(ptr as *mut __m128i, set.0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, set.1);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, set.2);
    _mm_storeu_si128(ptr.add(48) as *mut __m128i, set.3);
}
