use crate::sse::{_mm_exp_ps, _mm_log_ps, _mm_neg_ps, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn sse_color_to_sigmoidal(x: __m128) -> __m128 {
    let x = _mm_mul_ps(x, _mm_set1_ps(1f32 / 255f32));
    let negg = _mm_neg_ps(x);
    let den = _mm_add_ps(_mm_set1_ps(1f32), _mm_exp_ps(negg));
    let erase_nan_mask = _mm_cmpeq_ps(den, _mm_setzero_ps());
    let rcp = _mm_rcp_ps(den);
    return _mm_select_ps(erase_nan_mask, _mm_setzero_ps(), rcp);
}

#[inline(always)]
pub(crate) unsafe fn sse_sigmoidal_to_color(x: __m128) -> __m128 {
    let den = _mm_sub_ps(_mm_set1_ps(1f32), x);
    let zero_mask_1 = _mm_cmpeq_ps(den, _mm_setzero_ps());
    let k = _mm_mul_ps(x, _mm_rcp_ps(den));
    let zeros = _mm_setzero_ps();
    let zero_mask_2 = _mm_cmple_ps(k, zeros);
    let ln = _mm_log_ps(k);
    let rs = _mm_select_ps(_mm_and_ps(zero_mask_1, zero_mask_2), zeros, ln);
    return rs;
}

#[inline(always)]
pub(crate) unsafe fn sse_rgb_to_sigmoidal(
    r: __m128i,
    g: __m128i,
    b: __m128i,
) -> (__m128, __m128, __m128) {
    let sr = sse_color_to_sigmoidal(_mm_cvtepi32_ps(r));
    let sg = sse_color_to_sigmoidal(_mm_cvtepi32_ps(g));
    let sb = sse_color_to_sigmoidal(_mm_cvtepi32_ps(b));
    (sr, sg, sb)
}

#[inline(always)]
pub(crate) unsafe fn sse_sigmoidal_to_rgb(
    sr: __m128,
    sg: __m128,
    sb: __m128,
) -> (__m128i, __m128i, __m128i) {
    let sr = sse_sigmoidal_to_color(sr);
    let sg = sse_sigmoidal_to_color(sg);
    let sb = sse_sigmoidal_to_color(sb);
    let color_scale = _mm_set1_ps(255f32);
    let r = _mm_mul_ps(sr, color_scale);
    let g = _mm_mul_ps(sg, color_scale);
    let b = _mm_mul_ps(sb, color_scale);
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    (
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b)),
    )
}
