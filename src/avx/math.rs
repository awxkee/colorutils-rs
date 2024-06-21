use crate::sse::{_mm_mulsign_ps, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn _mm256_cube_ps(x: __m256) -> __m256 {
    _mm256_mul_ps(_mm256_mul_ps(x, x), x)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub unsafe fn _mm256_prefer_fma_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    return _mm256_add_ps(_mm256_mul_ps(b, c), a);
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn _mm256_prefer_fma_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    return _mm256_fmadd_ps(b, c, a);
}

#[inline(always)]
unsafe fn _mm256_taylorpoly_ps(
    x: __m256,
    poly0: __m256,
    poly1: __m256,
    poly2: __m256,
    poly3: __m256,
    poly4: __m256,
    poly5: __m256,
    poly6: __m256,
    poly7: __m256,
) -> __m256 {
    let a = _mm256_prefer_fma_ps(poly0, poly4, x);
    let b = _mm256_prefer_fma_ps(poly2, poly6, x);
    let c = _mm256_prefer_fma_ps(poly1, poly5, x);
    let d = _mm256_prefer_fma_ps(poly3, poly7, x);
    let x2 = _mm256_mul_ps(x, x);
    let x4 = _mm256_mul_ps(x2, x2);
    let res = _mm256_prefer_fma_ps(
        _mm256_prefer_fma_ps(a, b, x2),
        _mm256_prefer_fma_ps(c, d, x2),
        x4,
    );
    return res;
}

#[inline(always)]
pub unsafe fn _mm256_log_ps<const HANDLE_NAN: bool>(v: __m256) -> __m256 {
    let zeros = _mm256_setzero_ps();
    let nan_mask = _mm256_cmp_ps::<_CMP_LE_OS>(v, zeros);
    let const_ln127 = _mm256_set1_epi32(127); // 127
    let const_ln2 = _mm256_set1_ps(std::f32::consts::LN_2); // ln(2)

    // Extract exponent
    let m = _mm256_sub_epi32(_mm256_srli_epi32::<23>(_mm256_castps_si256(v)), const_ln127);
    let val = _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_castps_si256(v),
        _mm256_slli_epi32::<23>(m),
    ));

    let mut poly = _mm256_taylorpoly_ps(
        val,
        _mm256_set1_ps(-2.29561495781f32),
        _mm256_set1_ps(-2.47071170807f32),
        _mm256_set1_ps(-5.68692588806f32),
        _mm256_set1_ps(-0.165253549814f32),
        _mm256_set1_ps(5.17591238022f32),
        _mm256_set1_ps(0.844007015228f32),
        _mm256_set1_ps(4.58445882797f32),
        _mm256_set1_ps(0.0141278216615f32),
    );

    poly = _mm256_prefer_fma_ps(poly, _mm256_cvtepi32_ps(m), const_ln2);

    if HANDLE_NAN {
        poly = _mm256_select_ps(nan_mask, _mm256_set1_ps(-f32::INFINITY), poly);
    } else {
        poly = _mm256_select_ps(nan_mask, zeros, poly);
    }

    poly
}

#[inline(always)]
pub unsafe fn _mm256_select_ps(mask: __m256, true_vals: __m256, false_vals: __m256) -> __m256 {
    _mm256_blendv_ps(false_vals, true_vals, mask)
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_selecti_ps(mask: __m256i, true_vals: __m256, false_vals: __m256) -> __m256 {
    _mm256_blendv_ps(false_vals, true_vals, _mm256_castsi256_ps(mask))
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_select_si256(
    mask: __m256i,
    true_vals: __m256i,
    false_vals: __m256i,
) -> __m256i {
    _mm256_or_si256(
        _mm256_and_si256(mask, true_vals),
        _mm256_andnot_si256(mask, false_vals),
    )
}

#[inline(always)]
pub unsafe fn _mm256_exp_ps(x: __m256) -> __m256 {
    _mm256_exp_ps_ulp_1_5::<false>(x)
}

#[inline(always)]
pub unsafe fn _mm256_exp_ps_ulp_1_5<const HANDLE_NAN: bool>(x: __m256) -> __m256 {
    let c1 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffff6)); // x^1: 0x1.ffffecp-1f
    let c2 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffedb)); // x^2: 0x1.fffdb6p-2f
    let c3 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aaf33)); // x^3: 0x1.555e66p-3f
    let c4 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9f17)); // x^4: 0x1.573e2ep-5f
    let c5 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3c072010)); // x^5: 0x1.0e4020p-7f

    let shift = _mm256_castsi256_ps(_mm256_set1_epi32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    let inv_ln2 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    let neg_ln2_hi = _mm256_castsi256_ps(_mm256_set1_epi32(-1087278592i32)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    let neg_ln2_lo = _mm256_castsi256_ps(_mm256_set1_epi32(-1245725042i32)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

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
    let z = _mm256_prefer_fma_ps(shift, x, inv_ln2);
    let n = _mm256_sub_ps(z, shift);
    let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_castps_si256(z))); // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy beyond FP32.
    // This outperforms longer Taylor series (3-4 tabs) both in terms of accuracy and performance.
    let r_hi = _mm256_prefer_fma_ps(x, n, neg_ln2_hi);
    let r = _mm256_prefer_fma_ps(r_hi, n, neg_ln2_lo);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    let r2 = _mm256_mul_ps(r, r);

    let p1 = _mm256_mul_ps(c1, r);
    let p23 = _mm256_prefer_fma_ps(c2, c3, r);
    let p45 = _mm256_prefer_fma_ps(c4, c5, r);
    let p2345 = _mm256_prefer_fma_ps(p23, p45, r2);
    let p12345 = _mm256_prefer_fma_ps(p1, p2345, r2);

    let mut poly = _mm256_prefer_fma_ps(scale, p12345, scale);

    if HANDLE_NAN {
        let inf = _mm256_set1_ps(f32::INFINITY);
        let max_input = _mm256_set1_ps(88.37f32); // Approximately ln(2^127.5)
        let zero = _mm256_set1_ps(0f32);
        let min_input = _mm256_set1_ps(-86.64f32); // Approximately ln(2^-125)
                                                   // Handle underflow and overflow.
        poly = _mm256_select_ps(_mm256_cmp_ps::<_CMP_LT_OS>(x, min_input), zero, poly);
        poly = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(x, max_input), inf, poly);
    }

    return poly;
}

#[inline(always)]
unsafe fn _mm256_exp_ps_impl<const PROCESS_NAN: bool>(x: __m256) -> __m256 {
    let l2e = _mm256_set1_ps(std::f32::consts::LOG2_E); /* log2(e) */
    let c0 = _mm256_set1_ps(0.3371894346f32);
    let c1 = _mm256_set1_ps(0.657636276f32);
    let c2 = _mm256_set1_ps(1.00172476f32);

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    let t = _mm256_mul_ps(x, l2e); /* t = log2(e) * x */
    let e = _mm256_floor_ps(t); /* floor(t) */
    let i = _mm256_cvtps_epi32(e); /* (int)floor(t) */
    let f = _mm256_sub_ps(t, e); /* f = t - floor(t) */
    let mut p = c0; /* c0 */
    p = _mm256_prefer_fma_ps(c1, p, f); /* c0 * f + c1 */
    p = _mm256_prefer_fma_ps(c2, p, f); /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    let j = _mm256_slli_epi32::<23>(i); /* i << 23 */
    let r = _mm256_castsi256_ps(_mm256_add_epi32(j, _mm256_castps_si256(p))); /* r = p * 2^i*/
    if PROCESS_NAN {
        let inf = _mm256_set1_ps(f32::INFINITY);
        let max_input = _mm256_set1_ps(88.72283f32); // Approximately ln(2^127.5)
        let min_input = _mm256_set1_ps(-87.33654f32); // Approximately ln(2^-125)
        let poly = _mm256_select_ps(
            _mm256_cmp_ps::<_CMP_LT_OS>(x, min_input),
            _mm256_setzero_ps(),
            r,
        );
        let poly = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(x, max_input), inf, poly);
        return poly;
    } else {
        return r;
    }
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_pow_ps(x: __m256, n: __m256) -> __m256 {
    _mm256_exp_ps(_mm256_mul_ps(n, _mm256_log_ps(x)))
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_pow_n_ps(x: __m256, n: f32) -> __m256 {
    _mm256_exp_ps(_mm256_mul_ps(_mm256_set1_ps(n), _mm256_log_ps(x)))
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_signbit_ps(f: __m256) -> __m256i {
    return _mm256_and_si256(
        _mm256_castps_si256(f),
        _mm256_castps_si256(_mm256_set1_ps(-0.0f32)),
    );
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_mulsign_ps(x: __m256, y: __m256) -> __m256 {
    return _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_castps_si256(x),
        _mm256_signbit_ps(y),
    ));
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_pow2i_ps(q: __m256i) -> __m256 {
    return _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        q,
        _mm256_set1_epi32(0x7f),
    )));
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_vldexp2_ps(d: __m256, e: __m256i) -> __m256 {
    return _mm256_mul_ps(
        _mm256_mul_ps(d, _mm256_pow2i_ps(_mm256_srli_epi32::<1>(e))),
        _mm256_pow2i_ps(_mm256_sub_epi32(e, _mm256_srli_epi32::<1>(e))),
    );
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm256_vilogbk_ps(d: __m256) -> __m256i {
    let o = _mm256_cmp_ps::<_CMP_LT_OS>(d, _mm256_set1_ps(5.421010862427522E-20f32));
    let d = _mm256_select_ps(
        o,
        _mm256_mul_ps(_mm256_set1_ps(1.8446744073709552E19f32), d),
        d,
    );
    let q = _mm256_and_si256(
        _mm256_srli_epi32::<23>(_mm256_castps_si256(d)),
        _mm256_set1_epi32(0xff),
    );
    let q = _mm256_sub_epi32(
        q,
        _mm256_select_si256(
            _mm256_castps_si256(o),
            _mm256_set1_epi32(64 + 0x7f),
            _mm256_set1_epi32(0x7f),
        ),
    );
    return q;
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn _mm256_fmaf_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_prefer_fma_ps(c, b, a)
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn _mm256_abs_ps(x: __m256) -> __m256 {
    let sign_mask = _mm256_set1_ps(-0f32);
    return _mm256_andnot_ps(sign_mask, x);
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn _mm256_neg_epi32(x: __m256i) -> __m256i {
    let high = _mm256_set1_epi32(0i32);
    return _mm256_sub_epi32(high, x);
}

#[inline(always)]
/// This is Cube Root using Pow functions,
/// it is also precise however due to of inexact nature of power 1/3 result slightly differ
/// from real cbrt with about ULP 3-4, but this is almost 2 times faster than cbrt with real ULP 3.5
pub unsafe fn _mm256_cbrt_ps(d: __m256) -> __m256 {
    _mm_cbrtq_f32_ulp2::<false>(d)
}

#[inline(always)]
pub unsafe fn _mm256_cmpge_epi32(a: __m256i, b: __m256i) -> __m256i {
    let gt = _mm256_cmpgt_epi32(a, b);
    let eq = _mm256_cmpeq_epi32(a, b);
    return _mm256_or_si256(gt, eq);
}

#[inline(always)]
pub unsafe fn _mm256_cmplt_epi32(a: __m256i, b: __m256i) -> __m256i {
    return _mm256_cmpgt_epi32(b, a);
}

#[inline(always)]
/// Precise version of Cube Root with ULP 2
pub unsafe fn _mm_cbrtq_f32_ulp2<const HANDLE_NAN: bool>(x: __m256) -> __m256 {
    let x1p24 = _mm256_castsi256_ps(_mm256_set1_epi32(0x4b800000)); // 0x1p24f === 2 ^ 24

    let mut ui = _mm256_cvtps_epi32(x);
    let hx = _mm256_and_si256(ui, _mm256_set1_epi32(0x7fffffff));

    let nan_mask = _mm256_cmpge_epi32(hx, _mm256_set1_epi32(0x7f800000));
    let is_zero_mask = _mm256_cmpeq_epi32(hx, _mm256_setzero_si256());

    let lo_mask = _mm256_cmplt_epi32(hx, _mm256_set1_epi32(0x00800000));
    let hi_ui_f = _mm256_castps_si256(_mm256_mul_ps(x, x1p24));
    let mut lo_hx = _mm256_and_si256(hi_ui_f, _mm256_set1_epi32(0x7fffffff));
    let recpreq_3 = _mm256_set1_ps(1f32 / 3f32);
    lo_hx = _mm256_add_epi32(
        _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(lo_hx), recpreq_3)),
        _mm256_set1_epi32(642849266),
    );
    let hi_hx = _mm256_add_epi32(
        _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(hx), recpreq_3)),
        _mm256_set1_epi32(709958130),
    );
    let hx = _mm256_select_si256(lo_mask, lo_hx, hi_hx);

    ui = _mm256_select_si256(lo_mask, hi_ui_f, ui);
    ui = _mm256_and_si256(ui, _mm256_set1_epi32(-2147483648i32));
    ui = _mm256_or_si256(ui, hx);

    let mut t = _mm256_castsi256_ps(ui);
    let mut r = _mm256_mul_ps(_mm256_mul_ps(t, t), t);

    let sum_x = _mm256_add_ps(x, x);

    t = _mm256_mul_ps(
        _mm256_div_ps(
            _mm256_add_ps(sum_x, r),
            _mm256_add_ps(_mm256_add_ps(r, r), x),
        ),
        t,
    );

    r = _mm256_mul_ps(_mm256_mul_ps(t, t), t);
    t = _mm256_mul_ps(
        _mm256_div_ps(
            _mm256_add_ps(sum_x, r),
            _mm256_add_ps(_mm256_add_ps(r, r), x),
        ),
        t,
    );
    if HANDLE_NAN {
        t = _mm256_selecti_ps(nan_mask, _mm256_set1_ps(f32::NAN), t);
        t = _mm256_selecti_ps(is_zero_mask, _mm256_setzero_ps(), t);
    }
    t
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
/// Precise version of Cube Root, ULP 3.5
pub unsafe fn _mm256_cbrt_ps_ulp35(d: __m256) -> __m256 {
    let mut q = _mm256_set1_ps(1f32);
    let e = _mm256_add_epi32(_mm256_vilogbk_ps(_mm256_abs_ps(d)), _mm256_set1_epi32(1));
    let mut d = _mm256_vldexp2_ps(d, _mm256_neg_epi32(e));

    let t = _mm256_add_ps(_mm256_cvtepi32_ps(e), _mm256_set1_ps(6144f32));
    let qu = _mm256_cvttps_epi32(_mm256_mul_ps(t, _mm256_set1_ps(1.0f32 / 3.0f32)));
    let re = _mm256_cvttps_epi32(_mm256_sub_ps(
        t,
        _mm256_mul_ps(_mm256_cvtepi32_ps(qu), _mm256_set1_ps(3f32)),
    ));

    q = _mm256_selecti_ps(
        _mm256_cmpeq_epi32(re, _mm256_set1_epi32(1)),
        _mm256_set1_ps(1.2599210498948731647672106f32),
        q,
    );
    q = _mm256_selecti_ps(
        _mm256_cmpeq_epi32(re, _mm256_set1_epi32(2)),
        _mm256_set1_ps(1.5874010519681994747517056f32),
        q,
    );
    q = _mm256_vldexp2_ps(q, _mm256_sub_epi32(qu, _mm256_set1_epi32(2048)));
    q = _mm256_mulsign_ps(q, d);
    d = _mm256_abs_ps(d);

    let mut x = _mm256_set1_ps(-0.601564466953277587890625f32);
    x = _mm256_fmaf_ps(x, d, _mm256_set1_ps(2.8208892345428466796875f32));
    x = _mm256_fmaf_ps(x, d, _mm256_set1_ps(-5.532182216644287109375f32));
    x = _mm256_fmaf_ps(x, d, _mm256_set1_ps(5.898262500762939453125f32));
    x = _mm256_fmaf_ps(x, d, _mm256_set1_ps(-3.8095417022705078125f32));
    x = _mm256_fmaf_ps(x, d, _mm256_set1_ps(2.2241256237030029296875f32));

    let mut y = _mm256_mul_ps(_mm256_mul_ps(d, x), x);
    y = _mm256_mul_ps(
        _mm256_sub_ps(
            y,
            _mm256_mul_ps(
                _mm256_mul_ps(y, _mm256_set1_ps(2.0f32 / 3.0f32)),
                _mm256_fmaf_ps(y, x, _mm256_set1_ps(-1.0f32)),
            ),
        ),
        q,
    );
    return y;
}

#[inline(always)]
pub unsafe fn _mm256_color_matrix_ps(
    r: __m256,
    g: __m256,
    b: __m256,
    c1: __m256,
    c2: __m256,
    c3: __m256,
    c4: __m256,
    c5: __m256,
    c6: __m256,
    c7: __m256,
    c8: __m256,
    c9: __m256,
) -> (__m256, __m256, __m256) {
    let new_r = _mm256_prefer_fma_ps(_mm256_prefer_fma_ps(_mm256_mul_ps(g, c2), b, c3), r, c1);
    let new_g = _mm256_prefer_fma_ps(_mm256_prefer_fma_ps(_mm256_mul_ps(g, c5), b, c6), r, c4);
    let new_b = _mm256_prefer_fma_ps(_mm256_prefer_fma_ps(_mm256_mul_ps(g, c8), b, c9), r, c7);
    (new_r, new_g, new_b)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_neg_ps(x: __m256) -> __m256 {
    let high = _mm256_set1_ps(0f32);
    return _mm256_sub_ps(high, x);
}

#[inline(always)]
pub unsafe fn _mm256_is_infinity(d: __m256) -> __m256 {
    return _mm256_cmp_ps::<_CMP_EQ_OS>(_mm256_abs_ps(d), _mm256_set1_ps(f32::INFINITY));
}

#[inline(always)]
pub unsafe fn _mm256_cos_ps(d: __m256) -> __m256 {
    let mut q = _mm256_cvtps_epi32(_mm256_sub_ps(
        _mm256_mul_ps(d, _mm256_set1_ps(std::f32::consts::FRAC_1_PI)),
        _mm256_set1_ps(0.5f32),
    ));

    q = _mm256_add_epi32(_mm256_add_epi32(q, q), _mm256_set1_epi32(1));

    let mut u = _mm256_cvtepi32_ps(q);
    let mut d = _mm256_fmaf_ps(u, _mm256_set1_ps(-0.78515625f32 * 2f32), d);
    d = _mm256_fmaf_ps(u, _mm256_set1_ps(-0.00024187564849853515625f32 * 2f32), d);
    d = _mm256_fmaf_ps(u, _mm256_set1_ps(-3.7747668102383613586e-08f32 * 2f32), d);
    d = _mm256_fmaf_ps(u, _mm256_set1_ps(-1.2816720341285448015e-12f32 * 2f32), d);

    let s = _mm256_mul_ps(d, d);

    d = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(
            _mm256_cmpeq_epi32(
                _mm256_and_si256(q, _mm256_set1_epi32(2)),
                _mm256_set1_epi32(0),
            ),
            _mm256_castps_si256(_mm256_set1_ps(-0.0f32)),
        ),
        _mm256_castps_si256(d),
    ));

    u = _mm256_set1_ps(2.6083159809786593541503e-06f32);
    u = _mm256_fmaf_ps(u, s, _mm256_set1_ps(-0.0001981069071916863322258f32));
    u = _mm256_fmaf_ps(u, s, _mm256_set1_ps(0.00833307858556509017944336f32));
    u = _mm256_fmaf_ps(u, s, _mm256_set1_ps(-0.166666597127914428710938f32));

    u = _mm256_fmaf_ps(s, _mm256_mul_ps(u, d), d);

    u = _mm256_or_ps(_mm256_is_infinity(d), u);

    return u;
}

#[inline(always)]
pub unsafe fn _mm256_hypot_ps(x: __m256, y: __m256) -> __m256 {
    let xp2 = _mm256_mul_ps(x, x);
    let yp2 = _mm256_mul_ps(y, y);
    let z = _mm256_add_ps(xp2, yp2);
    return _mm256_sqrt_ps(z);
}

#[inline(always)]
pub unsafe fn _mm256_poly4_ps(
    x: __m256,
    x2: __m256,
    c3: __m256,
    c2: __m256,
    c1: __m256,
    c0: __m256,
) -> __m256 {
    _mm256_fmaf_ps(x2, _mm256_fmaf_ps(x, c3, c2), _mm256_fmaf_ps(x, c1, c0))
}

#[inline(always)]
pub unsafe fn _mm256_poly8q_ps(
    x: __m256,
    x2: __m256,
    x4: __m256,
    c7: __m256,
    c6: __m256,
    c5: __m256,
    c4: __m256,
    c3: __m256,
    c2: __m256,
    c1: __m256,
    c0: __m256,
) -> __m256 {
    _mm256_fmaf_ps(
        x4,
        _mm256_poly4_ps(x, x2, c7, c6, c5, c4),
        _mm256_poly4_ps(x, x2, c3, c2, c1, c0),
    )
}

#[inline(always)]
unsafe fn _mm256_atan2q_ps_impl(y: __m256, x: __m256) -> __m256 {
    let q = _mm256_select_si256(
        _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LT_OS>(x, _mm256_setzero_ps())),
        _mm256_set1_epi32(-2),
        _mm256_set1_epi32(0),
    );
    let x = _mm256_abs_ps(x);
    let is_y_more_than_x = _mm256_cmp_ps::<_CMP_GT_OS>(y, x);
    let t = _mm256_select_ps(is_y_more_than_x, x, _mm256_setzero_ps());
    let x = _mm256_select_ps(is_y_more_than_x, y, x);
    let y = _mm256_select_ps(is_y_more_than_x, _mm256_neg_ps(t), y);
    let q = _mm256_select_si256(
        _mm256_castps_si256(is_y_more_than_x),
        _mm256_add_epi32(q, _mm256_set1_epi32(1)),
        q,
    );
    let s = _mm256_div_ps(y, x);
    let t = _mm256_mul_ps(s, s);
    let t2 = _mm256_mul_ps(t, t);
    let t4 = _mm256_mul_ps(t2, t2);
    let poly = _mm256_poly8q_ps(
        t,
        t2,
        t4,
        _mm256_set1_ps(0.00282363896258175373077393f32),
        _mm256_set1_ps(-0.0159569028764963150024414f32),
        _mm256_set1_ps(0.0425049886107444763183594f32),
        _mm256_set1_ps(-0.0748900920152664184570312f32),
        _mm256_set1_ps(0.106347933411598205566406f32),
        _mm256_set1_ps(-0.142027363181114196777344f32),
        _mm256_set1_ps(0.199926957488059997558594f32),
        _mm256_set1_ps(-0.333331018686294555664062f32),
    );
    let t = _mm256_prefer_fma_ps(s, _mm256_mul_ps(poly, t), s);
    let t = _mm256_prefer_fma_ps(
        t,
        _mm256_cvtepi32_ps(q),
        _mm256_set1_ps(std::f32::consts::FRAC_PI_2),
    );
    t
}

#[inline(always)]
pub unsafe fn _mm256_atan2_ps(y: __m256, x: __m256) -> __m256 {
    let r = _mm256_atan2q_ps_impl(_mm256_abs_ps(y), x);
    let mut r = _mm256_mulsign_ps(r, x);
    let zeros = _mm256_setzero_ps();
    let y_zero_mask = _mm256_cmp_ps::<_CMP_EQ_OS>(y, zeros);
    r = _mm256_select_ps(
        _mm256_cmp_ps::<_CMP_EQ_OS>(x, zeros),
        _mm256_set1_ps(std::f32::consts::FRAC_PI_2),
        r,
    );
    r = _mm256_select_ps(y_zero_mask, zeros, r);
    _mm256_mulsign_ps(r, y)
}

#[inline(always)]
pub unsafe fn _mm256_sin_ps(val: __m256) -> __m256 {
    let pi_v = _mm256_set1_ps(std::f32::consts::PI);
    let pio2_v = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
    let ipi_v = _mm256_set1_ps(std::f32::consts::FRAC_1_PI);

    //Find positive or negative
    let c_v = _mm256_abs_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(val, ipi_v)));
    let sign_v = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OS>(val, _mm256_setzero_ps()));
    let odd_v = _mm256_and_si256(c_v, _mm256_set1_epi32(1));

    let neg_v = _mm256_xor_si256(odd_v, sign_v);

    //Modulus a - (n * int(a*(1/n)))
    let mut ma = _mm256_sub_ps(
        _mm256_abs_ps(val),
        _mm256_mul_ps(pi_v, _mm256_cvtepi32_ps(c_v)),
    );
    let reb_v = _mm256_cmp_ps::<_CMP_GE_OS>(ma, pio2_v);

    //Rebase a between 0 and pi/2
    ma = _mm256_select_ps(reb_v, _mm256_sub_ps(pi_v, ma), ma);

    //Taylor series
    let ma2 = _mm256_mul_ps(ma, ma);

    //2nd elem: x^3 / 3!
    let mut elem = _mm256_mul_ps(_mm256_mul_ps(ma, ma2), _mm256_set1_ps(0.166666666666f32));
    let mut res = _mm256_sub_ps(ma, elem);

    //3rd elem: x^5 / 5!
    elem = _mm256_mul_ps(_mm256_mul_ps(elem, ma2), _mm256_set1_ps(0.05f32));
    res = _mm256_add_ps(res, elem);

    //4th elem: x^7 / 7!float32x2_t vsin_f32(float32x2_t val)
    elem = _mm256_mul_ps(_mm256_mul_ps(elem, ma2), _mm256_set1_ps(0.023809523810f32));
    res = _mm256_sub_ps(res, elem);

    //5th elem: x^9 / 9!
    elem = _mm256_mul_ps(_mm256_mul_ps(elem, ma2), _mm256_set1_ps(0.013888888889f32));
    res = _mm256_add_ps(res, elem);

    //Change of sign
    let neg_v = _mm256_slli_epi32::<31>(neg_v);
    res = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(res), neg_v));
    return res;
}
