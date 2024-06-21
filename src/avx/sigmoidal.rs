use crate::avx::{_mm256_exp_ps, _mm256_log_ps, _mm256_neg_ps, _mm256_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn avx_color_to_sigmoidal(x: __m256) -> __m256 {
    let x = _mm256_mul_ps(x, _mm256_set1_ps(1f32 / 255f32));
    let negg = _mm256_neg_ps(x);
    let den = _mm256_add_ps(_mm256_set1_ps(1f32), _mm256_exp_ps(negg));
    let erase_nan_mask = _mm256_cmp_ps::<_CMP_EQ_OS>(den, _mm256_setzero_ps());
    let rcp = _mm256_rcp_ps(den);
    return _mm256_select_ps(erase_nan_mask, _mm256_setzero_ps(), rcp);
}

#[inline(always)]
pub(crate) unsafe fn avx_sigmoidal_to_color(x: __m256) -> __m256 {
    let den = _mm256_sub_ps(_mm256_set1_ps(1f32), x);
    let zero_mask_1 = _mm256_cmp_ps::<_CMP_EQ_OS>(den, _mm256_setzero_ps());
    let k = _mm256_mul_ps(x, _mm256_rcp_ps(den));
    let zeros = _mm256_setzero_ps();
    let zero_mask_2 = _mm256_cmp_ps::<_CMP_LT_OS>(k, zeros);
    let ln = _mm256_log_ps::<false>(k);
    let rs = _mm256_select_ps(_mm256_and_ps(zero_mask_1, zero_mask_2), zeros, ln);
    return rs;
}

#[inline(always)]
pub(crate) unsafe fn avx_rgb_to_sigmoidal(
    r: __m256i,
    g: __m256i,
    b: __m256i,
) -> (__m256, __m256, __m256) {
    let sr = avx_color_to_sigmoidal(_mm256_cvtepi32_ps(r));
    let sg = avx_color_to_sigmoidal(_mm256_cvtepi32_ps(g));
    let sb = avx_color_to_sigmoidal(_mm256_cvtepi32_ps(b));
    (sr, sg, sb)
}

#[inline(always)]
pub(crate) unsafe fn avx_sigmoidal_to_rgb(
    sr: __m256,
    sg: __m256,
    sb: __m256,
) -> (__m256i, __m256i, __m256i) {
    let sr = avx_sigmoidal_to_color(sr);
    let sg = avx_sigmoidal_to_color(sg);
    let sb = avx_sigmoidal_to_color(sb);
    let color_scale = _mm256_set1_ps(255f32);
    let r = _mm256_mul_ps(sr, color_scale);
    let g = _mm256_mul_ps(sg, color_scale);
    let b = _mm256_mul_ps(sb, color_scale);
    (
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(r)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(g)),
        _mm256_cvtps_epi32(_mm256_round_ps::<0>(b)),
    )
}
