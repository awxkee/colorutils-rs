#[allow(unused_imports)]
use crate::gamma_curves::TransferFunction;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx_math::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx_srgb_from_linear(linear: __m256) -> __m256 {
    let low_cut_off = _mm256_set1_ps(0.0030412825601275209f32);
    let mask = _mm256_cmp_ps::<_CMP_GE_OS>(linear, low_cut_off);

    let mut low = linear;
    let mut high = linear;
    low = _mm256_mul_ps(low, _mm256_set1_ps(12.92f32));

    high = _mm256_sub_ps(
        _mm256_mul_ps(
            _mm256_pow_n_ps(high, 1.0f32 / 2.4f32),
            _mm256_set1_ps(1.0550107189475866f32),
        ),
        _mm256_set1_ps(0.0550107189475866f32),
    );
    return _mm256_select_ps(mask, high, low);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx_srgb_to_linear(gamma: __m256) -> __m256 {
    let low_cut_off = _mm256_set1_ps(12.92f32 * 0.0030412825601275209f32);
    let mask = _mm256_cmp_ps::<_CMP_GE_OS>(gamma, low_cut_off);

    let mut low = gamma;
    let high = _mm256_pow_n_ps(
        _mm256_mul_ps(
            _mm256_add_ps(gamma, _mm256_set1_ps(0.0550107189475866f32)),
            _mm256_set1_ps(1f32 / 1.0550107189475866f32),
        ),
        2.4f32,
    );
    low = _mm256_mul_ps(low, _mm256_set1_ps(1f32 / 12.92f32));
    return _mm256_select_ps(mask, high, low);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx_rec709_from_linear(linear: __m256) -> __m256 {
    let low_cut_off = _mm256_set1_ps(0.018053968510807f32);
    let mask = _mm256_cmp_ps::<_CMP_GE_OS>(linear, low_cut_off);

    let mut low = linear;
    let mut high = linear;
    low = _mm256_mul_ps(low, _mm256_set1_ps(4.5f32));

    high = _mm256_sub_ps(
        _mm256_mul_ps(
            _mm256_pow_n_ps(high, 0.45f32),
            _mm256_set1_ps(1.09929682680944f32),
        ),
        _mm256_set1_ps(0.09929682680944f32),
    );
    return _mm256_select_ps(mask, high, low);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx_rec709_to_linear(linear: __m256) -> __m256 {
    let low_cut_off = _mm256_set1_ps(4.5f32 * 0.018053968510807f32);
    let mask = _mm256_cmp_ps::<_CMP_GE_OS>(linear, low_cut_off);

    let mut low = linear;
    let high = _mm256_pow_n_ps(
        _mm256_mul_ps(
            _mm256_add_ps(linear, _mm256_set1_ps(0.09929682680944f32)),
            _mm256_set1_ps(1f32 / 1.09929682680944f32),
        ),
        1.0f32 / 0.45f32,
    );
    low = _mm256_mul_ps(low, _mm256_set1_ps(1f32 / 4.5f32));
    return _mm256_select_ps(mask, high, low);
}