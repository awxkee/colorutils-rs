#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn vfmodq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let dividend_vec = a;
    let divisor_vec = b;
    let division = vmulq_f32(dividend_vec, vrecpeq_f32(divisor_vec));
    let int_part = vcvtq_f32_s32(vcvtq_s32_f32(division));
    let product = vmulq_f32(int_part, divisor_vec);
    let remainder = vsubq_f32(dividend_vec, product);
    remainder
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn prefer_vfmaq_f32(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    #[cfg(target_arch = "aarch64")]
    {
        return vfmaq_f32(a, b, c);
    }
    #[cfg(target_arch = "arm")]
    {
        return vmlaq_f32(a, b, c);
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn vtaylor_polyq_f32(
    x: float32x4_t,
    poly0: float32x4_t,
    poly1: float32x4_t,
    poly2: float32x4_t,
    poly3: float32x4_t,
    poly4: float32x4_t,
    poly5: float32x4_t,
    poly6: float32x4_t,
    poly7: float32x4_t,
) -> float32x4_t {
    let a = prefer_vfmaq_f32(poly0, poly4, x);
    let b = prefer_vfmaq_f32(poly2, poly6, x);
    let c = prefer_vfmaq_f32(poly1, poly5, x);
    let d = prefer_vfmaq_f32(poly3, poly7, x);
    let x2 = vmulq_f32(x, x);
    let x4 = vmulq_f32(x2, x2);
    let res = prefer_vfmaq_f32(prefer_vfmaq_f32(a, b, x2), prefer_vfmaq_f32(c, d, x2), x4);
    return res;
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vrintq_s32(d: float32x4_t) -> int32x4_t {
    return vcvtq_s32_f32(vaddq_f32(
        d,
        vreinterpretq_f32_u32(vorrq_u32(
            vandq_u32(
                vreinterpretq_u32_f32(d),
                vreinterpretq_u32_f32(vdupq_n_f32(-0.0f32)),
            ),
            vreinterpretq_u32_f32(vdupq_n_f32(0.5f32)),
        )),
    ));
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vfloorq_f32(x: float32x4_t) -> float32x4_t {
    let const_1 = vdupq_n_f32(1f32);

    let z = vcvtq_s32_f32(x);
    let r = vcvtq_f32_s32(z);

    return vbslq_f32(vcgtq_f32(r, x), vsubq_f32(r, const_1), r);
}

#[inline(always)]
pub unsafe fn vexpq_f32(x: float32x4_t) -> float32x4_t {
    vexpq_f32_ulp1_5::<false>(x)
}

#[inline(always)]
#[allow(dead_code)]
/// Low precision exp(x), ULP ~5
unsafe fn vexpq_f32_impl_ulp5<const PROCESS_NAN: bool>(x: float32x4_t) -> float32x4_t {
    let l2e = vdupq_n_f32(std::f32::consts::LOG2_E); /* log2(e) */
    let c0 = vdupq_n_f32(0.3371894346f32);
    let c1 = vdupq_n_f32(0.657636276f32);
    let c2 = vdupq_n_f32(1.00172476f32);

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    let t = vmulq_f32(x, l2e); /* t = log2(e) * x */
    let e = vfloorq_f32(t); /* floor(t) */
    let i = vcvtq_s32_f32(e); /* (int)floor(t) */
    let f = vsubq_f32(t, e); /* f = t - floor(t) */
    let mut p = c0; /* c0 */
    p = prefer_vfmaq_f32(c1, p, f); /* c0 * f + c1 */
    p = prefer_vfmaq_f32(c2, p, f); /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    let j = vshlq_n_s32::<23>(i); /* i << 23 */
    let r = vreinterpretq_f32_s32(vaddq_s32(j, vreinterpretq_s32_f32(p))); /* r = p * 2^i*/
    if PROCESS_NAN {
        let inf = vdupq_n_f32(f32::INFINITY);
        let max_input = vdupq_n_f32(88.72283f32); // Approximately ln(2^127.5)
        let min_input = vdupq_n_f32(-87.33654f32); // Approximately ln(2^-125)
        let poly = vbslq_f32(vcltq_f32(x, min_input), vdupq_n_f32(0f32), r);
        let poly = vbslq_f32(vcgtq_f32(x, max_input), inf, poly);
        return poly;
    } else {
        return r;
    }
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vexpq_f32_ulp1_5<const HANDLE_NAN: bool>(x: float32x4_t) -> float32x4_t {
    let c1 = vreinterpretq_f32_u32(vdupq_n_u32(0x3f7ffff6)); // x^1: 0x1.ffffecp-1f
    let c2 = vreinterpretq_f32_u32(vdupq_n_u32(0x3efffedb)); // x^2: 0x1.fffdb6p-2f
    let c3 = vreinterpretq_f32_u32(vdupq_n_u32(0x3e2aaf33)); // x^3: 0x1.555e66p-3f
    let c4 = vreinterpretq_f32_u32(vdupq_n_u32(0x3d2b9f17)); // x^4: 0x1.573e2ep-5f
    let c5 = vreinterpretq_f32_u32(vdupq_n_u32(0x3c072010)); // x^5: 0x1.0e4020p-7f

    let shift = vreinterpretq_f32_u32(vdupq_n_u32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    let inv_ln2 = vreinterpretq_f32_u32(vdupq_n_u32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    let neg_ln2_hi = vreinterpretq_f32_u32(vdupq_n_u32(0xbf317200)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    let neg_ln2_lo = vreinterpretq_f32_u32(vdupq_n_u32(0xb5bfbe8e)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

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
    let z = prefer_vfmaq_f32(shift, x, inv_ln2);
    let n = vsubq_f32(z, shift);
    let scale = vreinterpretq_f32_u32(vshlq_n_u32::<23>(vreinterpretq_u32_f32(z))); // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy beyond FP32.
    // This outperforms longer Taylor series (3-4 tabs) both in terms of accuracy and performance.
    let r_hi = prefer_vfmaq_f32(x, n, neg_ln2_hi);
    let r = prefer_vfmaq_f32(r_hi, n, neg_ln2_lo);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    let r2 = vmulq_f32(r, r);

    let p1 = vmulq_f32(c1, r);
    let p23 = prefer_vfmaq_f32(c2, c3, r);
    let p45 = prefer_vfmaq_f32(c4, c5, r);
    let p2345 = prefer_vfmaq_f32(p23, p45, r2);
    let p12345 = prefer_vfmaq_f32(p1, p2345, r2);

    let mut poly = prefer_vfmaq_f32(scale, p12345, scale);

    // Handle underflow and overflow.
    if HANDLE_NAN {
        let inf = vdupq_n_f32(f32::INFINITY);
        let max_input = vdupq_n_f32(88.37f32); // Approximately ln(2^127.5)
        let min_input = vdupq_n_f32(-86.64f32); // Approximately ln(2^-125)
        let zero = vdupq_n_f32(0f32);
        poly = vbslq_f32(vcltq_f32(x, min_input), zero, poly);
        poly = vbslq_f32(vcgtq_f32(x, max_input), inf, poly);
    }

    return poly;
}

#[inline(always)]
#[allow(dead_code)]
/// High precision exp. ULP = 1.0
pub unsafe fn vexpq_f32_ulp1(d: float32x4_t) -> float32x4_t {
    let q = vrintq_s32(vmulq_n_f32(d, std::f32::consts::LOG2_E));

    let mut s = vmlafq_f32(vcvtq_f32_s32(q), vdupq_n_f32(-std::f32::consts::LN_2), d);
    s = vmlafq_f32(
        vcvtq_f32_s32(q),
        vdupq_n_f32(-1.428606765330187045e-06f32),
        s,
    );

    let mut u = vdupq_n_f32(0.000198527617612853646278381f32);
    u = vmlafq_f32(u, s, vdupq_n_f32(0.00139304355252534151077271f32));
    u = vmlafq_f32(u, s, vdupq_n_f32(0.00833336077630519866943359f32));
    u = vmlafq_f32(u, s, vdupq_n_f32(0.0416664853692054748535156f32));
    u = vmlafq_f32(u, s, vdupq_n_f32(0.166666671633720397949219f32));
    u = vmlafq_f32(u, s, vdupq_n_f32(0.5f32));

    u = vaddq_f32(vdupq_n_f32(1.0f32), vmlafq_f32(vmulq_f32(s, s), u, s));

    u = vldexp2q_f32(u, q);

    u = vreinterpretq_f32_u32(vbicq_u32(
        vreinterpretq_u32_f32(u),
        vcltq_f32(d, vdupq_n_f32(-104f32)),
    ));
    u = vbslq_f32(
        vcltq_f32(vdupq_n_f32(100f32), d),
        vdupq_n_f32(f32::INFINITY),
        u,
    );
    u
}

#[inline(always)]
pub unsafe fn vlogq_f32(x: float32x4_t) -> float32x4_t {
    let const_ln127 = vdupq_n_s32(127); // 127
    let const_ln2 = vdupq_n_f32(std::f32::consts::LN_2); // ln(2)

    // Extract exponent
    let m = vsubq_s32(
        vreinterpretq_s32_u32(vshrq_n_u32::<23>(vreinterpretq_u32_f32(x))),
        const_ln127,
    );
    let val = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32::<23>(m)));

    // Polynomial Approximation
    let mut poly = vtaylor_polyq_f32(
        val,
        vdupq_n_f32(-2.29561495781f32),
        vdupq_n_f32(-2.47071170807f32),
        vdupq_n_f32(-5.68692588806f32),
        vdupq_n_f32(-0.165253549814f32),
        vdupq_n_f32(5.17591238022f32),
        vdupq_n_f32(0.844007015228f32),
        vdupq_n_f32(4.58445882797f32),
        vdupq_n_f32(0.0141278216615f32),
    );

    // Reconstruct
    poly = prefer_vfmaq_f32(poly, vcvtq_f32_s32(m), const_ln2);

    return poly;
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn visnanq_f32(x: float32x4_t) -> uint32x4_t {
    return vmvnq_u32(vceqq_f32(x, x));
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vispinfq_f32(d: float32x4_t) -> uint32x4_t {
    return vceqq_f32(d, vdupq_n_f32(f32::INFINITY));
}

#[inline(always)]
#[allow(dead_code)]
/// High precision log ULP = 3.5
pub unsafe fn vlogq_f32_ulp35(d: float32x4_t) -> float32x4_t {
    let o = vceqq_f32(d, vdupq_n_f32(f32::MIN));
    let m = (1i64 << 32i64) as f32;
    let d = vbslq_f32(o, vmulq_f32(d, vdupq_n_f32(m * m)), d);
    let e = vilogbk_vi2_vf(vmulq_f32(d, vdupq_n_f32(1.0f32 / 0.75f32)));
    let m = vldexp2q_f32(d, vnegq_s32(e));
    let e = vbslq_s32(o, vsubq_s32(e, vdupq_n_s32(64)), e);

    let mut x = vdivq_f32(
        vsubq_f32(m, vdupq_n_f32(1.0f32)),
        vaddq_f32(vdupq_n_f32(1.0f32), m),
    );
    let x2 = vmulq_f32(x, x);

    let mut t = vdupq_n_f32(0.2392828464508056640625f32);
    t = vmlafq_f32(t, x2, vdupq_n_f32(0.28518211841583251953125f32));
    t = vmlafq_f32(t, x2, vdupq_n_f32(0.400005877017974853515625f32));
    t = vmlafq_f32(t, x2, vdupq_n_f32(0.666666686534881591796875f32));
    t = vmlafq_f32(t, x2, vdupq_n_f32(2.0f32));

    x = vmlafq_f32(
        x,
        t,
        vmulq_f32(vdupq_n_f32(std::f32::consts::LN_2), vcvtq_f32_s32(e)),
    );
    x = vbslq_f32(vispinfq_f32(d), vdupq_n_f32(f32::NAN), x);
    x = vbslq_f32(
        vorrq_u32(vcltq_f32(d, vdupq_n_f32(0f32)), visnanq_f32(d)),
        vdupq_n_f32(f32::NAN),
        x,
    );
    x = vbslq_f32(vceqq_f32(d, vdupq_n_f32(0f32)), vdupq_n_f32(-f32::NAN), x);
    return x;
}

#[inline(always)]
pub unsafe fn vpowq_f32(val: float32x4_t, n: float32x4_t) -> float32x4_t {
    return vexpq_f32(vmulq_f32(n, vlogq_f32(val)));
}

#[inline(always)]
pub unsafe fn vpowq_n_f32(t: float32x4_t, power: f32) -> float32x4_t {
    return vpowq_f32(t, vdupq_n_f32(power));
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vilogbk_vi2_vf(d: float32x4_t) -> int32x4_t {
    let o = vcltq_f32(d, vdupq_n_f32(5.421010862427522E-20f32));
    let d = vbslq_f32(o, vmulq_f32(vdupq_n_f32(1.8446744073709552E19f32), d), d);
    let q = vandq_s32(
        vshrq_n_s32::<23>(vreinterpretq_s32_f32(d)),
        vdupq_n_s32(0xff),
    );
    let q = vsubq_s32(q, vbslq_s32(o, vdupq_n_s32(64 + 0x7f), vdupq_n_s32(0x7f)));
    return q;
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vpow2i(q: int32x4_t) -> float32x4_t {
    return vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(q, vdupq_n_s32(0x7f))));
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vldexp2q_f32(d: float32x4_t, e: int32x4_t) -> float32x4_t {
    return vmulq_f32(
        vmulq_f32(d, vpow2i(vshrq_n_s32::<1>(e))),
        vpow2i(vsubq_s32(e, vshrq_n_s32::<1>(e))),
    );
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vsignbit_vm_vf(f: float32x4_t) -> uint32x4_t {
    return vandq_u32(
        vreinterpretq_u32_f32(f),
        vreinterpretq_u32_f32(vdupq_n_f32(-0.0f32)),
    );
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vmulsignq_f32(x: float32x4_t, y: float32x4_t) -> float32x4_t {
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x), vsignbit_vm_vf(y)));
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn vmlafq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    prefer_vfmaq_f32(c, b, a)
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vcbrtq_f32(d: float32x4_t) -> float32x4_t {
    vcbrtq_f32_ulp2::<false>(d)
}

#[inline(always)]
/// Precise version of Cube Root with ULP 2
pub unsafe fn vcbrtq_f32_ulp2<const HANDLE_NAN: bool>(x: float32x4_t) -> float32x4_t {
    let x1p24 = vreinterpretq_f32_u32(vdupq_n_u32(0x4b800000)); // 0x1p24f === 2 ^ 24

    let mut ui = vreinterpretq_u32_f32(x);
    let hx = vandq_u32(ui, vdupq_n_u32(0x7fffffff));

    let nan_mask = vcgeq_u32(hx, vdupq_n_u32(0x7f800000));
    let is_zero_mask = vceqzq_u32(hx);

    let lo_mask = vcltq_u32(hx, vdupq_n_u32(0x00800000));
    let hi_ui_f = vreinterpretq_u32_f32(vmulq_f32(x, x1p24));
    let mut lo_hx = vandq_u32(hi_ui_f, vdupq_n_u32(0x7fffffff));
    lo_hx = vaddq_u32(
        vcvtq_u32_f32(vmulq_n_f32(vcvtq_f32_u32(lo_hx), 1f32 / 3f32)),
        vdupq_n_u32(642849266),
    );
    let hi_hx = vaddq_u32(
        vcvtq_u32_f32(vmulq_n_f32(vcvtq_f32_u32(hx), 1f32 / 3f32)),
        vdupq_n_u32(709958130),
    );
    let hx = vbslq_u32(lo_mask, lo_hx, hi_hx);

    ui = vbslq_u32(lo_mask, hi_ui_f, ui);
    ui = vandq_u32(ui, vdupq_n_u32(0x80000000));
    ui = vorrq_u32(ui, hx);

    let mut t = vreinterpretq_f32_u32(ui);
    let mut r = vmulq_f32(vmulq_f32(t, t), t);

    let sum_x = vaddq_f32(x, x);

    t = vmulq_f32(
        vdivq_f32(vaddq_f32(sum_x, r), vaddq_f32(vaddq_f32(r, r), x)),
        t,
    );

    r = vmulq_f32(vmulq_f32(t, t), t);
    t = vmulq_f32(
        vdivq_f32(vaddq_f32(sum_x, r), vaddq_f32(vaddq_f32(r, r), x)),
        t,
    );
    if HANDLE_NAN {
        t = vbslq_f32(nan_mask, vdupq_n_f32(f32::NAN), t);
        t = vbslq_f32(is_zero_mask, vdupq_n_f32(0f32), t);
    }
    t
}

#[inline(always)]
#[allow(dead_code)]
/// Precise version of Cube Root with ULP 3.5
pub unsafe fn vcbrtq_f32_ulp35(d: float32x4_t) -> float32x4_t {
    let mut q = vdupq_n_f32(1f32);
    let e = vaddq_s32(vilogbk_vi2_vf(vabsq_f32(d)), vdupq_n_s32(1));
    let mut d = vldexp2q_f32(d, vnegq_s32(e));

    let t = vaddq_f32(vcvtq_f32_s32(e), vdupq_n_f32(6144f32));
    let qu = vcvtq_s32_f32(vmulq_n_f32(t, 1.0f32 / 3.0f32));
    let re = vcvtq_s32_f32(vsubq_f32(t, vmulq_n_f32(vcvtq_f32_s32(qu), 3f32)));

    q = vbslq_f32(
        vceqq_s32(re, vdupq_n_s32(1)),
        vdupq_n_f32(1.2599210498948731647672106f32),
        q,
    );
    q = vbslq_f32(
        vceqq_s32(re, vdupq_n_s32(2)),
        vdupq_n_f32(1.5874010519681994747517056f32),
        q,
    );
    q = vldexp2q_f32(q, vsubq_s32(qu, vdupq_n_s32(2048)));

    q = vmulsignq_f32(q, d);
    d = vabsq_f32(d);

    let mut x = vdupq_n_f32(-0.601564466953277587890625f32);
    x = vmlafq_f32(x, d, vdupq_n_f32(2.8208892345428466796875f32));
    x = vmlafq_f32(x, d, vdupq_n_f32(-5.532182216644287109375f32));
    x = vmlafq_f32(x, d, vdupq_n_f32(5.898262500762939453125f32));
    x = vmlafq_f32(x, d, vdupq_n_f32(-3.8095417022705078125f32));
    x = vmlafq_f32(x, d, vdupq_n_f32(2.2241256237030029296875f32));

    let mut y = vmulq_f32(vmulq_f32(d, x), x);
    y = vmulq_f32(
        vsubq_f32(
            y,
            vmulq_f32(
                vmulq_n_f32(y, 2.0f32 / 3.0f32),
                vmlafq_f32(y, x, vdupq_n_f32(-1.0f32)),
            ),
        ),
        q,
    );
    return y;
}

#[inline(always)]
pub unsafe fn vcolorq_matrix_f32(
    r: float32x4_t,
    g: float32x4_t,
    b: float32x4_t,
    c1: float32x4_t,
    c2: float32x4_t,
    c3: float32x4_t,
    c4: float32x4_t,
    c5: float32x4_t,
    c6: float32x4_t,
    c7: float32x4_t,
    c8: float32x4_t,
    c9: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let new_r = prefer_vfmaq_f32(prefer_vfmaq_f32(vmulq_f32(g, c2), b, c3), r, c1);
    let new_g = prefer_vfmaq_f32(prefer_vfmaq_f32(vmulq_f32(g, c5), b, c6), r, c4);
    let new_b = prefer_vfmaq_f32(prefer_vfmaq_f32(vmulq_f32(g, c8), b, c9), r, c7);
    (new_r, new_g, new_b)
}
