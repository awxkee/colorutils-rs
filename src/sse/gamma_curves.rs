/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#![allow(dead_code)]
use crate::gamma_curves::TransferFunction;
use crate::sse::*;
use erydanos::_mm_pow_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn sse_srgb_from_linear(linear: __m128) -> __m128 {
    let linear = _mm_max_ps(linear, _mm_setzero_ps());
    let linear = _mm_min_ps(linear, _mm_set1_ps(1f32));
    let low_cut_off = _mm_set1_ps(0.0030412825601275209f32);
    let mask = _mm_cmpge_ps(linear, low_cut_off);

    let mut low = linear;
    let mut high = linear;
    low = _mm_mul_ps(low, _mm_set1_ps(12.92f32));

    high = _mm_sub_ps(
        _mm_mul_ps(
            _mm_pow_n_ps(high, 1.0f32 / 2.4f32),
            _mm_set1_ps(1.0550107189475866f32),
        ),
        _mm_set1_ps(0.0550107189475866f32),
    );
    _mm_select_ps(mask, high, low)
}

#[inline(always)]
pub unsafe fn sse_srgb_to_linear(gamma: __m128) -> __m128 {
    let gamma = _mm_max_ps(gamma, _mm_setzero_ps());
    let gamma = _mm_min_ps(gamma, _mm_set1_ps(1f32));
    let low_cut_off = _mm_set1_ps(12.92f32 * 0.0030412825601275209f32);
    let mask = _mm_cmpge_ps(gamma, low_cut_off);

    let mut low = gamma;
    let high = _mm_pow_n_ps(
        _mm_mul_ps(
            _mm_add_ps(gamma, _mm_set1_ps(0.0550107189475866f32)),
            _mm_set1_ps(1f32 / 1.0550107189475866f32),
        ),
        2.4f32,
    );
    low = _mm_mul_ps(low, _mm_set1_ps(1f32 / 12.92f32));
    _mm_select_ps(mask, high, low)
}

#[inline(always)]
pub unsafe fn sse_rec709_from_linear(linear: __m128) -> __m128 {
    let linear = _mm_max_ps(linear, _mm_setzero_ps());
    let linear = _mm_min_ps(linear, _mm_set1_ps(1f32));
    let low_cut_off = _mm_set1_ps(0.018053968510807f32);
    let mask = _mm_cmpge_ps(linear, low_cut_off);

    let mut low = linear;
    let mut high = linear;
    low = _mm_mul_ps(low, _mm_set1_ps(4.5f32));

    high = _mm_sub_ps(
        _mm_mul_ps(
            _mm_pow_n_ps(high, 0.45f32),
            _mm_set1_ps(1.09929682680944f32),
        ),
        _mm_set1_ps(0.09929682680944f32),
    );
    _mm_select_ps(mask, high, low)
}

#[inline(always)]
pub unsafe fn sse_rec709_to_linear(gamma: __m128) -> __m128 {
    let gamma = _mm_max_ps(gamma, _mm_setzero_ps());
    let gamma = _mm_min_ps(gamma, _mm_set1_ps(1f32));
    let low_cut_off = _mm_set1_ps(4.5f32 * 0.018053968510807f32);
    let mask = _mm_cmpge_ps(gamma, low_cut_off);

    let mut low = gamma;
    let high = _mm_pow_n_ps(
        _mm_mul_ps(
            _mm_add_ps(gamma, _mm_set1_ps(0.09929682680944f32)),
            _mm_set1_ps(1f32 / 1.09929682680944f32),
        ),
        1.0f32 / 0.45f32,
    );
    low = _mm_mul_ps(low, _mm_set1_ps(1f32 / 4.5f32));
    _mm_select_ps(mask, high, low)
}

#[inline(always)]
pub unsafe fn sse_pure_gamma(gamma: __m128, value: f32) -> __m128 {
    let zeros = _mm_setzero_ps();
    let zero_mask = _mm_cmple_ps(gamma, zeros);
    let ones = _mm_set1_ps(1f32);
    let ones_mask = _mm_cmpge_ps(gamma, ones);
    let mut rs = _mm_pow_n_ps(gamma, value);
    rs = _mm_select_ps(zero_mask, zeros, rs);
    _mm_select_ps(ones_mask, ones, rs)
}

#[inline(always)]
pub unsafe fn sse_smpte428_from_linear(linear: __m128) -> __m128 {
    const POWER_VALUE: f32 = 1.0f32 / 2.6f32;
    _mm_pow_ps(
        _mm_mul_ps(
            _mm_max_ps(linear, _mm_setzero_ps()),
            _mm_set1_ps(0.91655527974030934f32),
        ),
        _mm_set1_ps(POWER_VALUE),
    )
}

#[inline(always)]
pub unsafe fn sse_smpte428_to_linear(gamma: __m128) -> __m128 {
    const SCALE: f32 = 1. / 0.91655527974030934f32;
    _mm_mul_ps(
        _mm_pow_ps(_mm_max_ps(gamma, _mm_setzero_ps()), _mm_set1_ps(2.6f32)),
        _mm_set1_ps(SCALE),
    )
}

#[inline(always)]
pub unsafe fn sse_gamma2p2_to_linear(gamma: __m128) -> __m128 {
    sse_pure_gamma(gamma, 2.2f32)
}

#[inline(always)]
pub unsafe fn sse_gamma2p8_to_linear(gamma: __m128) -> __m128 {
    sse_pure_gamma(gamma, 2.8f32)
}

#[inline(always)]
pub unsafe fn sse_gamma2p2_from_linear(linear: __m128) -> __m128 {
    sse_pure_gamma(linear, 1f32 / 2.2f32)
}

#[inline(always)]
pub unsafe fn sse_gamma2p8_from_linear(linear: __m128) -> __m128 {
    sse_pure_gamma(linear, 1f32 / 2.8f32)
}