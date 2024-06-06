#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::x86_64_simd_support::shuffle;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_promote_i16_toi32(s: __m128i) -> __m128i {
    _mm_cvtepi16_epi32(_mm_srli_si128::<8>(s))
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_interleave_even(x: __m128i) -> __m128i {
    #[rustfmt::skip]
        let shuffle = _mm_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6,
                                    8, 8, 10, 10, 12, 12, 14, 14);
    let new_lane = _mm_shuffle_epi8(x, shuffle);
    return new_lane;
}


#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_transpose_x4(
    r: __m128,
    g: __m128,
    b: __m128,
    a: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let t0 = _mm_castps_si128(_mm_unpacklo_ps(r, g));
    let t1 = _mm_castps_si128(_mm_unpacklo_ps(b, a));
    let t2 = _mm_castps_si128(_mm_unpackhi_ps(r, g));
    let t3 = _mm_castps_si128(_mm_unpackhi_ps(b, a));

    let row1 = _mm_castsi128_ps(_mm_unpacklo_epi64(t0, t1));
    let row2 = _mm_castsi128_ps(_mm_unpackhi_epi64(t0, t1));
    let row3 = _mm_castsi128_ps(_mm_unpacklo_epi64(t2, t3));
    let row4 = _mm_castsi128_ps(_mm_unpackhi_epi64(t2, t3));

    (row1, row2, row3, row4)
}


#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_store_rgba(ptr: *mut u8, r: __m128i, g: __m128i, b: __m128i, a: __m128i) {
    let (row1, row2, row3, row4) = sse_interleave_rgba(r, g, b, a);
    _mm_storeu_si128(ptr as *mut __m128i, row1);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, row2);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, row3);
    _mm_storeu_si128(ptr.add(48) as *mut __m128i, row4);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_store_rgb_u8(ptr: *mut u8, r: __m128i, g: __m128i, b: __m128i) {
    let (v0, v1, v2) = sse_interleave_rgb(r, g, b);
    _mm_storeu_si128(ptr as *mut __m128i, v0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, v1);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, v2);
}


#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_div_by255(v: __m128i) -> __m128i {
    let rounding = _mm_set1_epi16(1 << 7);
    let x = _mm_adds_epi16(v, rounding);
    let multiplier = _mm_set1_epi16(-32640);
    let r = _mm_mulhi_epu16(x, multiplier);
    return _mm_srli_epi16::<7>(r);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
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