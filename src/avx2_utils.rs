#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx2_interleave_rgb_ps(a: __m256, b: __m256, c: __m256) -> (__m256, __m256, __m256) {
    let b0 = _mm256_shuffle_epi32::<0x6c>(_mm256_castps_si256(a));
    let g0 = _mm256_shuffle_epi32::<0xb1>(_mm256_castps_si256(b));
    let r0 = _mm256_shuffle_epi32::<0xc6>(_mm256_castps_si256(c));

    let p0 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(b0, g0), r0);
    let p1 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(g0, r0), b0);
    let p2 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(r0, b0), g0);

    let bgr0 = _mm256_permute2x128_si256::<32>(p0, p1);
    //let bgr1 = p2;
    let bgr2 = _mm256_permute2x128_si256::<49>(p0, p1);
    (_mm256_castsi256_ps(bgr0), _mm256_castsi256_ps(p2), _mm256_castsi256_ps(bgr2))
}