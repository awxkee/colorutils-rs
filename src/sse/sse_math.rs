#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub unsafe fn _mm_cube_ps(x: __m128) -> __m128 {
    _mm_mul_ps(_mm_mul_ps(x, x), x)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[cfg(not(target_feature = "fma"))]
#[inline]
#[allow(dead_code)]
pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    return _mm_add_ps(_mm_mul_ps(b, c), a);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[cfg(target_feature = "fma")]
#[inline]
#[allow(dead_code)]
pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    return _mm_fmadd_ps(b, c, a);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
#[allow(dead_code)]
unsafe fn _mm_taylorpoly_ps(
    x: __m128,
    poly0: __m128,
    poly1: __m128,
    poly2: __m128,
    poly3: __m128,
    poly4: __m128,
    poly5: __m128,
    poly6: __m128,
    poly7: __m128,
) -> __m128 {
    let a = _mm_prefer_fma_ps(poly0, poly4, x);
    let b = _mm_prefer_fma_ps(poly2, poly6, x);
    let c = _mm_prefer_fma_ps(poly1, poly5, x);
    let d = _mm_prefer_fma_ps(poly3, poly7, x);
    let x2 = _mm_mul_ps(x, x);
    let x4 = _mm_mul_ps(x2, x2);
    let res = _mm_prefer_fma_ps(_mm_prefer_fma_ps(a, b, x2), _mm_prefer_fma_ps(c, d, x2), x4);
    return res;
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_log_ps(v: __m128) -> __m128 {
    let const_ln127 = _mm_set1_epi32(127); // 127
    let const_ln2 = _mm_set1_ps(std::f32::consts::LN_2); // ln(2)

    // Extract exponent
    let m = _mm_sub_epi32(_mm_srli_epi32::<23>(_mm_castps_si128(v)), const_ln127);
    let val = _mm_castsi128_ps(_mm_sub_epi32(_mm_castps_si128(v), _mm_slli_epi32::<23>(m)));

    let mut poly = _mm_taylorpoly_ps(
        val,
        _mm_set1_ps(-2.29561495781f32),
        _mm_set1_ps(-2.47071170807f32),
        _mm_set1_ps(-5.68692588806f32),
        _mm_set1_ps(-0.165253549814f32),
        _mm_set1_ps(5.17591238022f32),
        _mm_set1_ps(0.844007015228f32),
        _mm_set1_ps(4.58445882797f32),
        _mm_set1_ps(0.0141278216615f32),
    );

    poly = _mm_prefer_fma_ps(poly, _mm_cvtepi32_ps(m), const_ln2);
    poly
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_select_ps(mask: __m128, true_vals: __m128, false_vals: __m128) -> __m128 {
    _mm_blendv_ps(false_vals, true_vals, mask)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_selecti_ps(mask: __m128i, true_vals: __m128, false_vals: __m128) -> __m128 {
    _mm_blendv_ps(false_vals, true_vals, _mm_castsi128_ps(mask))
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_select_si128(mask: __m128i, true_vals: __m128i, false_vals: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_and_si128(mask, true_vals),
        _mm_andnot_si128(mask, false_vals),
    )
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_exp_ps(x: __m128) -> __m128 {
    let c1 = _mm_castsi128_ps(_mm_set1_epi32(0x3f7ffff6)); // x^1: 0x1.ffffecp-1f
    let c2 = _mm_castsi128_ps(_mm_set1_epi32(0x3efffedb)); // x^2: 0x1.fffdb6p-2f
    let c3 = _mm_castsi128_ps(_mm_set1_epi32(0x3e2aaf33)); // x^3: 0x1.555e66p-3f
    let c4 = _mm_castsi128_ps(_mm_set1_epi32(0x3d2b9f17)); // x^4: 0x1.573e2ep-5f
    let c5 = _mm_castsi128_ps(_mm_set1_epi32(0x3c072010)); // x^5: 0x1.0e4020p-7f

    let shift = _mm_castsi128_ps(_mm_set1_epi32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    let inv_ln2 = _mm_castsi128_ps(_mm_set1_epi32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    let neg_ln2_hi = _mm_castsi128_ps(_mm_set1_epi32(-1087278592i32)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    let neg_ln2_lo = _mm_castsi128_ps(_mm_set1_epi32(-1245725042i32)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

    let inf = _mm_set1_ps(f32::INFINITY);
    let max_input = _mm_set1_ps(88.37f32); // Approximately ln(2^127.5)
    let zero = _mm_set1_ps(0f32);
    let min_input = _mm_set1_ps(-86.64f32); // Approximately ln(2^-125)

    // Range reduction:
    //   e^x = 2^n * e^r
    // where:
    //   n = floor(x / ln(2))
    //   r = x - n * ln(2)
    //
    // By adding x / ln(2) with 2^23 + 127 (shift):
    //   * As FP32 fraction part only has 23-bits, the addition of 2^23 + 127 forces decimal part
    //     of x / ln(2) out of the result. The integer part of x / ln(2) (i.e. n) + 127 will occupy
    //     the whole fraction part of z in FP32 format.
    //     Subtracting 2^23 + 127 (shift) from z will result in the integer part of x / ln(2)
    //     (i.e. n) because the decimal part has been pushed out and lost.
    //   * The addition of 127 makes the FP32 fraction part of z ready to be used as the exponent
    //     in FP32 format. Left shifting z by 23 bits will result in 2^n.
    let z = _mm_prefer_fma_ps(shift, x, inv_ln2);
    let n = _mm_sub_ps(z, shift);
    let scale = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_castps_si128(z))); // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy beyond FP32.
    // This outperforms longer Taylor series (3-4 tabs) both in terms of accuracy and performance.
    let r_hi = _mm_prefer_fma_ps(x, n, neg_ln2_hi);
    let r = _mm_prefer_fma_ps(r_hi, n, neg_ln2_lo);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    let r2 = _mm_mul_ps(r, r);

    let p1 = _mm_mul_ps(c1, r);
    let p23 = _mm_prefer_fma_ps(c2, c3, r);
    let p45 = _mm_prefer_fma_ps(c4, c5, r);
    let p2345 = _mm_prefer_fma_ps(p23, p45, r2);
    let p12345 = _mm_prefer_fma_ps(p1, p2345, r2);

    let mut poly = _mm_prefer_fma_ps(scale, p12345, scale);

    // Handle underflow and overflow.
    poly = _mm_select_ps(_mm_cmplt_ps(x, min_input), zero, poly);
    poly = _mm_select_ps(_mm_cmpgt_ps(x, max_input), inf, poly);

    return poly;
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_pow_ps(x: __m128, n: __m128) -> __m128 {
    _mm_exp_ps(_mm_mul_ps(n, _mm_log_ps(x)))
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_pow_n_ps(x: __m128, n: f32) -> __m128 {
    _mm_exp_ps(_mm_mul_ps(_mm_set1_ps(n), _mm_log_ps(x)))
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_signbit_ps(f: __m128) -> __m128i {
    return _mm_and_si128(_mm_castps_si128(f), _mm_castps_si128(_mm_set1_ps(-0.0f32)));
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_mulsign_ps(x: __m128, y: __m128) -> __m128 {
    return _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(x), _mm_signbit_ps(y)));
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_pow2i_ps(q: __m128i) -> __m128 {
    return _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(q, _mm_set1_epi32(0x7f))));
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_vldexp2_ps(d: __m128, e: __m128i) -> __m128 {
    return _mm_mul_ps(
        _mm_mul_ps(d, _mm_pow2i_ps(_mm_srli_epi32::<1>(e))),
        _mm_pow2i_ps(_mm_sub_epi32(e, _mm_srli_epi32::<1>(e))),
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_vilogbk_ps(d: __m128) -> __m128i {
    let o = _mm_cmplt_ps(d, _mm_set1_ps(5.421010862427522E-20f32));
    let d = _mm_select_ps(o, _mm_mul_ps(_mm_set1_ps(1.8446744073709552E19f32), d), d);
    let q = _mm_and_si128(
        _mm_srli_epi32::<23>(_mm_castps_si128(d)),
        _mm_set1_epi32(0xff),
    );
    let q = _mm_sub_epi32(
        q,
        _mm_select_si128(_mm_castps_si128(o), _mm_set1_epi32(64 + 0x7f), _mm_set1_epi32(0x7f)),
    );
    return q;
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn _mm_fmaf_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_prefer_fma_ps(c, b, a)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn _mm_abs_ps(x: __m128) -> __m128 {
    let sign_mask = _mm_set1_ps(-0f32);
    return _mm_andnot_ps(sign_mask, x);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn _mm_neg_epi32(x: __m128i) -> __m128i {
    let high = _mm_set1_epi32(0i32);
    return _mm_sub_epi32(high, x);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
/// This is Cube Root using Pow functions,
/// it is also precise however due to of inexact nature of power 1/3 result slightly differ
/// from real cbrt with about ULP 3-4, but this is almost 2 times faster than cbrt with real ULP 3.5
pub unsafe fn _mm_cbrt_ps(d: __m128) -> __m128 {
    _mm_pow_n_ps(d, 1f32 / 3f32)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
/// Precise version of Cube Root, ULP 3.5
pub unsafe fn _mm_cbrt_ps_ulp35(d: __m128) -> __m128 {
    let mut q = _mm_set1_ps(1f32);
    let e = _mm_add_epi32(_mm_vilogbk_ps(_mm_abs_ps(d)), _mm_set1_epi32(1));
    let mut d = _mm_vldexp2_ps(d, _mm_neg_epi32(e));

    let t = _mm_add_ps(_mm_cvtepi32_ps(e), _mm_set1_ps(6144f32));
    let qu = _mm_cvttps_epi32(_mm_mul_ps(t, _mm_set1_ps(1.0f32 / 3.0f32)));
    let re = _mm_cvttps_epi32(_mm_sub_ps(t, _mm_mul_ps(_mm_cvtepi32_ps(qu), _mm_set1_ps(3f32))));

    q = _mm_selecti_ps(
        _mm_cmpeq_epi32(re, _mm_set1_epi32(1)),
        _mm_set1_ps(1.2599210498948731647672106f32),
        q,
    );
    q = _mm_selecti_ps(
        _mm_cmpeq_epi32(re, _mm_set1_epi32(2)),
        _mm_set1_ps(1.5874010519681994747517056f32),
        q,
    );
    q = _mm_vldexp2_ps(q, _mm_sub_epi32(qu, _mm_set1_epi32(2048)));
    q = _mm_mulsign_ps(q, d);
    d = _mm_abs_ps(d);

    let mut x = _mm_set1_ps(-0.601564466953277587890625f32);
    x = _mm_fmaf_ps(x, d, _mm_set1_ps(2.8208892345428466796875f32));
    x = _mm_fmaf_ps(x, d, _mm_set1_ps(-5.532182216644287109375f32));
    x = _mm_fmaf_ps(x, d, _mm_set1_ps(5.898262500762939453125f32));
    x = _mm_fmaf_ps(x, d, _mm_set1_ps(-3.8095417022705078125f32));
    x = _mm_fmaf_ps(x, d, _mm_set1_ps(2.2241256237030029296875f32));

    let mut y = _mm_mul_ps(_mm_mul_ps(d, x), x);
    y = _mm_mul_ps(
        _mm_sub_ps(
            y,
            _mm_mul_ps(
                _mm_mul_ps(y, _mm_set1_ps(2.0f32 / 3.0f32)),
                _mm_fmaf_ps(y, x, _mm_set1_ps(-1.0f32)),
            ),
        ),
        q,
    );
    return y;
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_color_matrix_ps(
    r: __m128,
    g: __m128,
    b: __m128,
    c1: __m128,
    c2: __m128,
    c3: __m128,
    c4: __m128,
    c5: __m128,
    c6: __m128,
    c7: __m128,
    c8: __m128,
    c9: __m128,
) -> (__m128, __m128, __m128) {
    let new_r = _mm_prefer_fma_ps(_mm_prefer_fma_ps(_mm_mul_ps(g, c2), b, c3), r, c1);
    let new_g = _mm_prefer_fma_ps(_mm_prefer_fma_ps(_mm_mul_ps(g, c5), b, c6), r, c4);
    let new_b = _mm_prefer_fma_ps(_mm_prefer_fma_ps(_mm_mul_ps(g, c8), b, c9), r, c7);
    (new_r, new_g, new_b)
}
