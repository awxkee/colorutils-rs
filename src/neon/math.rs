/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use erydanos::vpowq_fast_f32;
use std::arch::aarch64::*;

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

#[inline(always)]
pub unsafe fn vpowjq_f32(val: float32x4_t, n: float32x4_t) -> float32x4_t {
    vpowq_fast_f32(val, n)
}

#[inline(always)]
pub unsafe fn vpowq_n_f32(t: float32x4_t, power: f32) -> float32x4_t {
    return vpowjq_f32(t, vdupq_n_f32(power));
}

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

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vpow2i(q: int32x4_t) -> float32x4_t {
    return vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(q, vdupq_n_s32(0x7f))));
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vldexp2q_f32(d: float32x4_t, e: int32x4_t) -> float32x4_t {
    return vmulq_f32(
        vmulq_f32(d, vpow2i(vshrq_n_s32::<1>(e))),
        vpow2i(vsubq_s32(e, vshrq_n_s32::<1>(e))),
    );
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vsignbit_vm_vf(f: float32x4_t) -> uint32x4_t {
    return vandq_u32(
        vreinterpretq_u32_f32(f),
        vreinterpretq_u32_f32(vdupq_n_f32(-0.0f32)),
    );
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn vmulsignq_f32(x: float32x4_t, y: float32x4_t) -> float32x4_t {
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x), vsignbit_vm_vf(y)));
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn vmlafq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    prefer_vfmaq_f32(c, b, a)
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
#[inline(always)]
pub(crate) unsafe fn vcubeq_f32(x: float32x4_t) -> float32x4_t {
    vmulq_f32(vmulq_f32(x, x), x)
}
