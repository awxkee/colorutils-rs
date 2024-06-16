use crate::avx::{_mm256_cube_ps, _mm256_prefer_fma_ps, _mm256_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::luv::{LUV_MULTIPLIER_INVERSE_Y, LUV_WHITE_U_PRIME, LUV_WHITE_V_PRIME};

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) unsafe fn avx_lab_to_xyz(l: __m256, a: __m256, b: __m256) -> (__m256, __m256, __m256) {
    let y = _mm256_mul_ps(
        _mm256_add_ps(l, _mm256_set1_ps(16f32)),
        _mm256_set1_ps(1f32 / 116f32),
    );
    let x = _mm256_add_ps(_mm256_mul_ps(a, _mm256_set1_ps(1f32 / 500f32)), y);
    let z = _mm256_sub_ps(y, _mm256_mul_ps(b, _mm256_set1_ps(1f32 / 200f32)));
    let x3 = _mm256_cube_ps(x);
    let y3 = _mm256_cube_ps(y);
    let z3 = _mm256_cube_ps(z);
    let kappa = _mm256_set1_ps(0.008856f32);
    let k_sub = _mm256_set1_ps(16f32 / 116f32);
    let mult_1 = _mm256_set1_ps(1f32 / 7.787f32);
    let low_x = _mm256_mul_ps(_mm256_sub_ps(x, k_sub), mult_1);
    let low_y = _mm256_mul_ps(_mm256_sub_ps(y, k_sub), mult_1);
    let low_z = _mm256_mul_ps(_mm256_sub_ps(z, k_sub), mult_1);

    let x = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(x3, kappa), x3, low_x);
    let y = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(y3, kappa), y3, low_y);
    let z = _mm256_select_ps(_mm256_cmp_ps::<_CMP_GT_OS>(z3, kappa), z3, low_z);
    let x = _mm256_mul_ps(x, _mm256_set1_ps(95.047f32 / 100f32));
    let z = _mm256_mul_ps(z, _mm256_set1_ps(108.883f32 / 100f32));
    (x, y, z)
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) unsafe fn avx_luv_to_xyz(l: __m256, u: __m256, v: __m256) -> (__m256, __m256, __m256) {
    let zeros = _mm256_setzero_ps();
    let zero_mask = _mm256_cmp_ps::<_CMP_EQ_OS>(l, zeros);
    let l13 = _mm256_rcp_ps(_mm256_mul_ps(l, _mm256_set1_ps(13f32)));
    let u = _mm256_prefer_fma_ps(_mm256_set1_ps(LUV_WHITE_U_PRIME), l13, u);
    let v = _mm256_prefer_fma_ps(_mm256_set1_ps(LUV_WHITE_V_PRIME), l13, v);
    let l_h = _mm256_mul_ps(
        _mm256_add_ps(l, _mm256_set1_ps(16f32)),
        _mm256_set1_ps(1f32 / 116f32),
    );
    let y_high = _mm256_mul_ps(_mm256_mul_ps(l_h, l_h), l_h);
    let y_low = _mm256_mul_ps(l, _mm256_set1_ps(LUV_MULTIPLIER_INVERSE_Y));
    let y = _mm256_select_ps(
        zero_mask,
        zeros,
        _mm256_select_ps(
            _mm256_cmp_ps::<_CMP_GT_OS>(l, _mm256_set1_ps(8f32)),
            y_high,
            y_low,
        ),
    );
    let zero_mask_2 = _mm256_cmp_ps::<_CMP_EQ_OS>(v, zeros);
    let den = _mm256_rcp_ps(_mm256_mul_ps(v, _mm256_set1_ps(4f32)));
    let mut x = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_mul_ps(y, u), den),
        _mm256_set1_ps(9f32),
    );
    x = _mm256_select_ps(zero_mask, zeros, x);
    x = _mm256_select_ps(zero_mask_2, zeros, x);
    let mut z = _mm256_mul_ps(
        _mm256_mul_ps(
            _mm256_prefer_fma_ps(
                _mm256_prefer_fma_ps(_mm256_set1_ps(12f32), _mm256_set1_ps(-3f32), u),
                v,
                _mm256_set1_ps(-20f32),
            ),
            y,
        ),
        den,
    );
    z = _mm256_select_ps(zero_mask, zeros, z);
    z = _mm256_select_ps(zero_mask_2, zeros, z);
    (x, y, z)
}
