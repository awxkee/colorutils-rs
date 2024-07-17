/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use erydanos::{vexpq_f32, vlnq_fast_f32};
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_color_to_sigmoidal(x: float32x4_t) -> float32x4_t {
    let x = vmulq_n_f32(x, 1f32 / 255f32);
    let negg = vnegq_f32(x);
    let den = vaddq_f32(vdupq_n_f32(1f32), vexpq_f32(negg));
    let erase_nan_mask = vceqzq_f32(den);
    let rcp = vrecpeq_f32(den);
    return vbslq_f32(erase_nan_mask, vdupq_n_f32(0f32), rcp);
}

#[inline(always)]
pub(crate) unsafe fn neon_sigmoidal_to_color(x: float32x4_t) -> float32x4_t {
    let den = vsubq_f32(vdupq_n_f32(1f32), x);
    let zero_mask_1 = vceqzq_f32(den);
    let k = vmulq_f32(x, vrecpeq_f32(den));
    let zeros = vdupq_n_f32(0f32);
    let zero_mask_2 = vcleq_f32(k, zeros);
    let ln = vlnq_fast_f32(k);
    let rs = vbslq_f32(vandq_u32(zero_mask_1, zero_mask_2), zeros, ln);
    return rs;
}

#[inline(always)]
pub(crate) unsafe fn neon_rgb_to_sigmoidal(
    r: uint32x4_t,
    g: uint32x4_t,
    b: uint32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let sr = neon_color_to_sigmoidal(vcvtq_f32_u32(r));
    let sg = neon_color_to_sigmoidal(vcvtq_f32_u32(g));
    let sb = neon_color_to_sigmoidal(vcvtq_f32_u32(b));
    (sr, sg, sb)
}

#[inline(always)]
pub(crate) unsafe fn neon_sigmoidal_to_rgb(
    sr: float32x4_t,
    sg: float32x4_t,
    sb: float32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t) {
    let sr = neon_sigmoidal_to_color(sr);
    let sg = neon_sigmoidal_to_color(sg);
    let sb = neon_sigmoidal_to_color(sb);
    let r = vmulq_n_f32(sr, 255f32);
    let g = vmulq_n_f32(sg, 255f32);
    let b = vmulq_n_f32(sb, 255f32);
    (vcvtaq_u32_f32(r), vcvtaq_u32_f32(g), vcvtaq_u32_f32(b))
}
