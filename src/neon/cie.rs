/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::luv::{
    LUV_CUTOFF_FORWARD_Y, LUV_MULTIPLIER_FORWARD_Y, LUV_MULTIPLIER_INVERSE_Y, LUV_WHITE_U_PRIME,
    LUV_WHITE_V_PRIME,
};
use crate::neon::math::{prefer_vfmaq_f32, vcolorq_matrix_f32, vcubeq_f32};
use erydanos::{vatan2q_f32, vcbrtq_fast_f32, vcosq_f32, vhypotq_fast_f32, vsinq_f32};
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_triple_to_xyz(
    r: uint32x4_t,
    g: uint32x4_t,
    b: uint32x4_t,
    c1: float32x4_t,
    c2: float32x4_t,
    c3: float32x4_t,
    c4: float32x4_t,
    c5: float32x4_t,
    c6: float32x4_t,
    c7: float32x4_t,
    c8: float32x4_t,
    c9: float32x4_t,
    transfer: &unsafe fn(float32x4_t) -> float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let r_f = vmulq_n_f32(vcvtq_f32_u32(r), 1f32 / 255f32);
    let g_f = vmulq_n_f32(vcvtq_f32_u32(g), 1f32 / 255f32);
    let b_f = vmulq_n_f32(vcvtq_f32_u32(b), 1f32 / 255f32);
    let r_linear = transfer(r_f);
    let g_linear = transfer(g_f);
    let b_linear = transfer(b_f);

    let (x, y, z) = vcolorq_matrix_f32(
        r_linear, g_linear, b_linear, c1, c2, c3, c4, c5, c6, c7, c8, c9,
    );
    (x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn neon_triple_to_luv(
    x: float32x4_t,
    y: float32x4_t,
    z: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let zeros = vdupq_n_f32(0f32);
    let den = prefer_vfmaq_f32(
        prefer_vfmaq_f32(x, z, vdupq_n_f32(3f32)),
        y,
        vdupq_n_f32(15f32),
    );
    let nan_mask = vceqzq_f32(den);
    let l_low_mask = vcltq_f32(y, vdupq_n_f32(LUV_CUTOFF_FORWARD_Y));
    let y_cbrt = vcbrtq_fast_f32(y);
    let l = vbslq_f32(
        l_low_mask,
        vmulq_n_f32(y, LUV_MULTIPLIER_FORWARD_Y),
        prefer_vfmaq_f32(vdupq_n_f32(-16f32), y_cbrt, vdupq_n_f32(116f32)),
    );
    let u_prime = vdivq_f32(vmulq_n_f32(x, 4f32), den);
    let v_prime = vdivq_f32(vmulq_n_f32(y, 9f32), den);
    let sub_u_prime = vsubq_f32(u_prime, vdupq_n_f32(LUV_WHITE_U_PRIME));
    let sub_v_prime = vsubq_f32(v_prime, vdupq_n_f32(LUV_WHITE_V_PRIME));
    let l13 = vmulq_n_f32(l, 13f32);
    let u = vbslq_f32(nan_mask, zeros, vmulq_f32(l13, sub_u_prime));
    let v = vbslq_f32(nan_mask, zeros, vmulq_f32(l13, sub_v_prime));
    (l, u, v)
}

#[inline(always)]
pub(crate) unsafe fn neon_triple_to_lab(
    x: float32x4_t,
    y: float32x4_t,
    z: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let x = vmulq_n_f32(x, 100f32 / 95.047f32);
    let z = vmulq_n_f32(z, 100f32 / 108.883f32);
    let cbrt_x = vcbrtq_fast_f32(x);
    let cbrt_y = vcbrtq_fast_f32(y);
    let cbrt_z = vcbrtq_fast_f32(z);
    let s_1 = vdupq_n_f32(16f32 / 116f32);
    let s_2 = vdupq_n_f32(7.787f32);
    let lower_x = prefer_vfmaq_f32(s_1, s_2, x);
    let lower_y = prefer_vfmaq_f32(s_1, s_2, y);
    let lower_z = prefer_vfmaq_f32(s_1, s_2, z);
    let kappa = vdupq_n_f32(0.008856f32);
    let x = vbslq_f32(vcgtq_f32(x, kappa), cbrt_x, lower_x);
    let y = vbslq_f32(vcgtq_f32(y, kappa), cbrt_y, lower_y);
    let z = vbslq_f32(vcgtq_f32(z, kappa), cbrt_z, lower_z);
    let l = prefer_vfmaq_f32(vdupq_n_f32(-16.0f32), y, vdupq_n_f32(116.0f32));
    let a = vmulq_n_f32(vsubq_f32(x, y), 500f32);
    let b = vmulq_n_f32(vsubq_f32(y, z), 200f32);
    (l, a, b)
}

#[inline(always)]
pub(crate) unsafe fn neon_triple_to_lch(
    x: float32x4_t,
    y: float32x4_t,
    z: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let (luv_l, luv_u, luv_v) = neon_triple_to_luv(x, y, z);
    let lch_c = vhypotq_fast_f32(luv_u, luv_v);
    let lch_h = vatan2q_f32(luv_v, luv_u);
    (luv_l, lch_c, lch_h)
}

#[inline(always)]
pub(crate) unsafe fn neon_luv_to_xyz(
    l: float32x4_t,
    u: float32x4_t,
    v: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let zero_mask = vclezq_f32(l);
    let zeros = vdupq_n_f32(0f32);
    let l13 = vrecpeq_f32(vmulq_n_f32(l, 13f32));
    let u = prefer_vfmaq_f32(vdupq_n_f32(LUV_WHITE_U_PRIME), l13, u);
    let v = prefer_vfmaq_f32(vdupq_n_f32(LUV_WHITE_V_PRIME), l13, v);
    let l_h = vmulq_n_f32(vaddq_f32(l, vdupq_n_f32(16f32)), 1f32 / 116f32);
    let y_high = vmulq_f32(vmulq_f32(l_h, l_h), l_h);
    let y_low = vmulq_n_f32(l, LUV_MULTIPLIER_INVERSE_Y);
    let y = vbslq_f32(
        zero_mask,
        zeros,
        vbslq_f32(vcgtq_f32(l, vdupq_n_f32(8f32)), y_high, y_low),
    );
    let zero_mask_2 = vclezq_f32(v);
    let den = vrecpeq_f32(vmulq_n_f32(v, 4f32));
    let mut x = vmulq_n_f32(vmulq_f32(vmulq_f32(y, u), den), 9f32);
    x = vbslq_f32(zero_mask, zeros, x);
    x = vbslq_f32(zero_mask_2, zeros, x);
    let mut z = vmulq_f32(
        vmulq_f32(
            prefer_vfmaq_f32(
                prefer_vfmaq_f32(vdupq_n_f32(12f32), vdupq_n_f32(-3f32), u),
                v,
                vdupq_n_f32(-20f32),
            ),
            y,
        ),
        den,
    );
    z = vbslq_f32(zero_mask, zeros, z);
    z = vbslq_f32(zero_mask_2, zeros, z);
    (x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn neon_lab_to_xyz(
    l: float32x4_t,
    a: float32x4_t,
    b: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let y = vmulq_n_f32(vaddq_f32(l, vdupq_n_f32(16f32)), 1f32 / 116f32);
    let x = vaddq_f32(vmulq_n_f32(a, 1f32 / 500f32), y);
    let z = vsubq_f32(y, vmulq_n_f32(b, 1f32 / 200f32));
    let x3 = vcubeq_f32(x);
    let y3 = vcubeq_f32(y);
    let z3 = vcubeq_f32(z);
    let kappa = vdupq_n_f32(0.008856f32);
    let k_sub = vdupq_n_f32(16f32 / 116f32);
    let low_x = vmulq_n_f32(vsubq_f32(x, k_sub), 1f32 / 7.787f32);
    let low_y = vmulq_n_f32(vsubq_f32(y, k_sub), 1f32 / 7.787f32);
    let low_z = vmulq_n_f32(vsubq_f32(z, k_sub), 1f32 / 7.787f32);

    let x = vbslq_f32(vcgtq_f32(x3, kappa), x3, low_x);
    let y = vbslq_f32(vcgtq_f32(y3, kappa), y3, low_y);
    let z = vbslq_f32(vcgtq_f32(z3, kappa), z3, low_z);
    let x = vmulq_n_f32(x, 95.047f32 / 100f32);
    let z = vmulq_n_f32(z, 108.883f32 / 100f32);
    (x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn neon_lch_to_xyz(
    l: float32x4_t,
    c: float32x4_t,
    h: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let u = vmulq_f32(c, vcosq_f32(h));
    let v = vmulq_f32(c, vsinq_f32(h));
    neon_luv_to_xyz(l, u, v)
}
