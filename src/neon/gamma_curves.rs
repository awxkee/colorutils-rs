/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#![allow(dead_code)]
use crate::neon::math::vpowq_n_f32;
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_srgb_from_linear(linear: float32x4_t) -> float32x4_t {
    let linear = vmaxq_f32(linear, vdupq_n_f32(0f32));
    let linear = vminq_f32(linear, vdupq_n_f32(1f32));
    let low_cut_off = vdupq_n_f32(0.0030412825601275209f32);
    let mask = vcgeq_f32(linear, low_cut_off);

    let mut low = linear;
    let mut high = linear;
    low = vmulq_n_f32(low, 12.92f32);

    high = vsubq_f32(
        vmulq_n_f32(vpowq_n_f32(high, 1.0f32 / 2.4f32), 1.0550107189475866f32),
        vdupq_n_f32(0.0550107189475866f32),
    );
    vbslq_f32(mask, high, low)
}

#[inline(always)]
pub unsafe fn neon_srgb_to_linear(gamma: float32x4_t) -> float32x4_t {
    let gamma = vmaxq_f32(gamma, vdupq_n_f32(0f32));
    let gamma = vminq_f32(gamma, vdupq_n_f32(1f32));
    let low_cut_off = vdupq_n_f32(12.92f32 * 0.0030412825601275209f32);
    let mask = vcgeq_f32(gamma, low_cut_off);

    let mut low = gamma;
    let high = vpowq_n_f32(
        vmulq_n_f32(
            vaddq_f32(gamma, vdupq_n_f32(0.0550107189475866f32)),
            1f32 / 1.0550107189475866f32,
        ),
        2.4f32,
    );
    low = vmulq_n_f32(low, 1f32 / 12.92f32);
    vbslq_f32(mask, high, low)
}

#[inline(always)]
pub unsafe fn neon_rec709_from_linear(linear: float32x4_t) -> float32x4_t {
    let linear = vmaxq_f32(linear, vdupq_n_f32(0f32));
    let linear = vminq_f32(linear, vdupq_n_f32(1f32));
    let low_cut_off = vdupq_n_f32(0.018053968510807f32);
    let mask = vcgeq_f32(linear, low_cut_off);

    let mut low = linear;
    let mut high = linear;
    low = vmulq_n_f32(low, 4.5f32);

    high = vsubq_f32(
        vmulq_n_f32(vpowq_n_f32(high, 0.45f32), 1.09929682680944f32),
        vdupq_n_f32(0.09929682680944f32),
    );
    vbslq_f32(mask, high, low)
}

#[inline(always)]
pub unsafe fn neon_rec709_to_linear(gamma: float32x4_t) -> float32x4_t {
    let gamma = vmaxq_f32(gamma, vdupq_n_f32(0f32));
    let gamma = vminq_f32(gamma, vdupq_n_f32(1f32));
    let low_cut_off = vdupq_n_f32(4.5f32 * 0.018053968510807f32);
    let mask = vcgeq_f32(gamma, low_cut_off);

    let mut low = gamma;
    let high = vpowq_n_f32(
        vmulq_n_f32(
            vaddq_f32(gamma, vdupq_n_f32(0.09929682680944f32)),
            1f32 / 1.09929682680944f32,
        ),
        1.0f32 / 0.45f32,
    );
    low = vmulq_n_f32(low, 1f32 / 4.5f32);
    vbslq_f32(mask, high, low)
}

#[inline(always)]
pub unsafe fn neon_pure_gamma_function(gamma: float32x4_t, gamma_constant: f32) -> float32x4_t {
    let zero_mask = vclezq_f32(gamma);
    let ones = vdupq_n_f32(1f32);
    let zeros = vdupq_n_f32(0f32);
    let ones_mask = vcgeq_f32(gamma, ones);
    let mut rs = vpowq_n_f32(gamma, gamma_constant);
    rs = vbslq_f32(zero_mask, zeros, rs);
    vbslq_f32(ones_mask, ones, rs)
}

#[inline(always)]
pub unsafe fn neon_smpte428_to_linear(gamma: float32x4_t) -> float32x4_t {
    const SCALE: f32 = 1. / 0.91655527974030934f32;
    vmulq_n_f32(
        vpowq_n_f32(vmaxq_f32(gamma, vdupq_n_f32(0.)), 2.6f32),
        SCALE,
    )
}

#[inline(always)]
pub unsafe fn neon_smpte428_from_linear(linear: float32x4_t) -> float32x4_t {
    const POWER_VALUE: f32 = 1.0f32 / 2.6f32;
    vpowq_n_f32(
        vmulq_n_f32(vmaxq_f32(linear, vdupq_n_f32(0.)), 0.91655527974030934f32),
        POWER_VALUE,
    )
}

#[inline(always)]
pub unsafe fn neon_gamma2p2_to_linear(gamma: float32x4_t) -> float32x4_t {
    neon_pure_gamma_function(gamma, 2.2f32)
}

#[inline(always)]
pub unsafe fn neon_gamma2p8_to_linear(gamma: float32x4_t) -> float32x4_t {
    neon_pure_gamma_function(gamma, 2.8f32)
}

#[inline(always)]
pub unsafe fn neon_gamma2p2_from_linear(linear: float32x4_t) -> float32x4_t {
    neon_pure_gamma_function(linear, 1f32 / 2.2f32)
}

#[inline(always)]
pub unsafe fn neon_gamma2p8_from_linear(linear: float32x4_t) -> float32x4_t {
    neon_pure_gamma_function(linear, 1f32 / 2.8f32)
}
