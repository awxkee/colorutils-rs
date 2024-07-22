/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use erydanos::vpowq_fast_f32;
use std::arch::aarch64::*;

#[inline(always)]
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
pub unsafe fn vpowjq_f32(val: float32x4_t, n: float32x4_t) -> float32x4_t {
    vpowq_fast_f32(val, n)
}

#[inline(always)]
pub unsafe fn vpowq_n_f32(t: float32x4_t, power: f32) -> float32x4_t {
    return vpowjq_f32(t, vdupq_n_f32(power));
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
