#[allow(unused_imports)]
use crate::gamma_curves::TransferFunction;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_srgb_from_linear(linear: __m128) -> __m128 {
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
    return _mm_select_ps(mask, high, low);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_srgb_to_linear(gamma: __m128) -> __m128 {
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
    return _mm_select_ps(mask, high, low);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_rec709_from_linear(linear: __m128) -> __m128 {
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
    return _mm_select_ps(mask, high, low);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_rec709_to_linear(linear: __m128) -> __m128 {
    let low_cut_off = _mm_set1_ps(4.5f32 * 0.018053968510807f32);
    let mask = _mm_cmpge_ps(linear, low_cut_off);

    let mut low = linear;
    let high = _mm_pow_n_ps(
        _mm_mul_ps(
            _mm_add_ps(linear, _mm_set1_ps(0.09929682680944f32)),
            _mm_set1_ps(1f32 / 1.09929682680944f32),
        ),
        1.0f32 / 0.45f32,
    );
    low = _mm_mul_ps(low, _mm_set1_ps(1f32 / 4.5f32));
    return _mm_select_ps(mask, high, low);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub unsafe fn get_sse_linear_transfer(
    transfer_function: TransferFunction,
) -> unsafe fn(__m128) -> __m128 {
    match transfer_function {
        TransferFunction::Srgb => sse_srgb_to_linear,
        TransferFunction::Rec709 => sse_rec709_to_linear,
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub unsafe fn get_sse_gamma_transfer(
    transfer_function: TransferFunction,
) -> unsafe fn(__m128) -> __m128 {
    match transfer_function {
        TransferFunction::Srgb => sse_srgb_from_linear,
        TransferFunction::Rec709 => sse_rec709_from_linear,
    }
}
