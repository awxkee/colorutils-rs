#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::luv::{LUV_MULTIPLIER_INVERSE_Y, LUV_WHITE_U_PRIME, LUV_WHITE_V_PRIME};
use crate::sse::{_mm_cube_ps, _mm_prefer_fma_ps, _mm_select_ps};

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) unsafe fn sse_lab_to_xyz(l: __m128, a: __m128, b: __m128) -> (__m128, __m128, __m128) {
    let y = _mm_mul_ps(
        _mm_add_ps(l, _mm_set1_ps(16f32)),
        _mm_set1_ps(1f32 / 116f32),
    );
    let x = _mm_add_ps(_mm_mul_ps(a, _mm_set1_ps(1f32 / 500f32)), y);
    let z = _mm_sub_ps(y, _mm_mul_ps(b, _mm_set1_ps(1f32 / 200f32)));
    let x3 = _mm_cube_ps(x);
    let y3 = _mm_cube_ps(y);
    let z3 = _mm_cube_ps(z);
    let kappa = _mm_set1_ps(0.008856f32);
    let k_sub = _mm_set1_ps(16f32 / 116f32);
    let mult_1 = _mm_set1_ps(1f32 / 7.787f32);
    let low_x = _mm_mul_ps(_mm_sub_ps(x, k_sub), mult_1);
    let low_y = _mm_mul_ps(_mm_sub_ps(y, k_sub), mult_1);
    let low_z = _mm_mul_ps(_mm_sub_ps(z, k_sub), mult_1);

    let x = _mm_select_ps(_mm_cmpgt_ps(x3, kappa), x3, low_x);
    let y = _mm_select_ps(_mm_cmpgt_ps(y3, kappa), y3, low_y);
    let z = _mm_select_ps(_mm_cmpgt_ps(z3, kappa), z3, low_z);
    let x = _mm_mul_ps(x, _mm_set1_ps(95.047f32 / 100f32));
    let z = _mm_mul_ps(z, _mm_set1_ps(108.883f32 / 100f32));
    (x, y, z)
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) unsafe fn sse_luv_to_xyz(l: __m128, u: __m128, v: __m128) -> (__m128, __m128, __m128) {
    let zeros = _mm_setzero_ps();
    let zero_mask = _mm_cmpeq_ps(l, zeros);
    let l13 = _mm_rcp_ps(_mm_mul_ps(l, _mm_set1_ps(13f32)));
    let u = _mm_prefer_fma_ps(_mm_set1_ps(LUV_WHITE_U_PRIME), l13, u);
    let v = _mm_prefer_fma_ps(_mm_set1_ps(LUV_WHITE_V_PRIME), l13, v);
    let l_h = _mm_mul_ps(
        _mm_add_ps(l, _mm_set1_ps(16f32)),
        _mm_set1_ps(1f32 / 116f32),
    );
    let y_high = _mm_mul_ps(_mm_mul_ps(l_h, l_h), l_h);
    let y_low = _mm_mul_ps(l, _mm_set1_ps(LUV_MULTIPLIER_INVERSE_Y));
    let y = _mm_select_ps(
        zero_mask,
        zeros,
        _mm_select_ps(_mm_cmpgt_ps(l, _mm_set1_ps(8f32)), y_high, y_low),
    );
    let zero_mask_2 = _mm_cmpeq_ps(v, zeros);
    let den = _mm_rcp_ps(_mm_mul_ps(v, _mm_set1_ps(4f32)));
    let mut x = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(y, u), den), _mm_set1_ps(9f32));
    x = _mm_select_ps(zero_mask, zeros, x);
    x = _mm_select_ps(zero_mask_2, zeros, x);
    let mut z = _mm_mul_ps(
        _mm_mul_ps(
            _mm_prefer_fma_ps(
                _mm_prefer_fma_ps(_mm_set1_ps(12f32), _mm_set1_ps(-3f32), u),
                v,
                _mm_set1_ps(-20f32),
            ),
            y,
        ),
        den,
    );
    z = _mm_select_ps(zero_mask, zeros, z);
    z = _mm_select_ps(zero_mask_2, zeros, z);
    (x, y, z)
}
