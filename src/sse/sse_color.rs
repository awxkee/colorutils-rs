#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::luv::{LUV_MULTIPLIER_INVERSE_Y, LUV_WHITE_U_PRIME, LUV_WHITE_V_PRIME};
use crate::sse::{_mm_abs_ps, _mm_cube_ps, _mm_fmod_ps, _mm_prefer_fma_ps, _mm_select_ps};

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

#[inline(always)]
pub unsafe fn sse_hsl_to_rgb(
    h: __m128,
    s: __m128,
    l: __m128,
    scale: __m128,
) -> (__m128i, __m128i, __m128i) {
    let s = _mm_mul_ps(s, scale);
    let l = _mm_mul_ps(l, scale);
    let ones = _mm_set1_ps(1f32);
    let twos = _mm_set1_ps(2f32);
    let c = _mm_mul_ps(
        _mm_sub_ps(ones, _mm_abs_ps(_mm_sub_ps(_mm_mul_ps(l, twos), ones))),
        s,
    );
    let x = _mm_mul_ps(
        _mm_sub_ps(
            ones,
            _mm_abs_ps(_mm_sub_ps(
                _mm_fmod_ps(_mm_mul_ps(h, _mm_set1_ps(1f32 / 60f32)), twos),
                ones,
            )),
        ),
        c,
    );

    let zeros = _mm_setzero_ps();
    let m = _mm_sub_ps(l, _mm_mul_ps(c, _mm_set1_ps(0.5f32)));
    let h_prime = h;
    let (mut r, mut g, mut b) = (zeros, zeros, zeros);

    let between_zero_and_one_mask = _mm_and_ps(
        _mm_cmpge_ps(h, zeros),
        _mm_cmplt_ps(h_prime, _mm_set1_ps(60f32)),
    );
    let between_one_and_two_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, _mm_set1_ps(60f32)),
        _mm_cmplt_ps(h_prime, _mm_set1_ps(120f32)),
    );
    let between_two_and_three_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, _mm_set1_ps(120f32)),
        _mm_cmplt_ps(h_prime, _mm_set1_ps(180f32)),
    );
    let between_three_and_four_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, _mm_set1_ps(180f32)),
        _mm_cmplt_ps(h_prime, _mm_set1_ps(240f32)),
    );
    let between_four_and_five_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, _mm_set1_ps(240f32)),
        _mm_cmplt_ps(h_prime, _mm_set1_ps(300f32)),
    );
    let between_five_and_six_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, _mm_set1_ps(300f32)),
        _mm_cmplt_ps(h_prime, _mm_set1_ps(360f32)),
    );
    // if h_prime >= 0f32 && h_prime < 1f32 {
    r = _mm_select_ps(between_zero_and_one_mask, c, r);
    g = _mm_select_ps(between_zero_and_one_mask, x, g);
    // if h_prime >= 1f32 && h_prime < 2f32 {
    r = _mm_select_ps(between_one_and_two_mask, x, r);
    g = _mm_select_ps(between_one_and_two_mask, c, g);
    // if h_prime >= 2f32 && h_prime < 3f32
    g = _mm_select_ps(between_two_and_three_mask, c, g);
    b = _mm_select_ps(between_two_and_three_mask, x, b);
    // if h_prime >= 3f32 && h_prime < 4f32 {
    g = _mm_select_ps(between_three_and_four_mask, x, g);
    b = _mm_select_ps(between_three_and_four_mask, c, b);
    // if h_prime >= 4f32 && h_prime < 5f32 {
    r = _mm_select_ps(between_four_and_five_mask, x, r);
    b = _mm_select_ps(between_four_and_five_mask, c, b);
    // if h_prime >= 5f32 && h_prime < 6f32 {
    r = _mm_select_ps(between_five_and_six_mask, c, r);
    b = _mm_select_ps(between_five_and_six_mask, x, b);
    r = _mm_add_ps(r, m);
    g = _mm_add_ps(g, m);
    b = _mm_add_ps(b, m);
    let rgb_scale = _mm_set1_ps(255f32);
    r = _mm_mul_ps(r, rgb_scale);
    g = _mm_mul_ps(g, rgb_scale);
    b = _mm_mul_ps(b, rgb_scale);
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    (
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b)),
    )
}

#[inline(always)]
pub unsafe fn sse_hsv_to_rgb(
    h: __m128,
    s: __m128,
    v: __m128,
    scale: __m128,
) -> (__m128i, __m128i, __m128i) {
    let s = _mm_mul_ps(s, scale);
    let v = _mm_mul_ps(v, scale);
    let c = _mm_mul_ps(s, v);
    let h_der = _mm_mul_ps(h, _mm_set1_ps(1f32 / 60f32));
    let six = _mm_set1_ps(6f32);
    let h_prime = _mm_fmod_ps(h_der, six);
    let ones = _mm_set1_ps(1f32);
    let x = _mm_mul_ps(
        _mm_sub_ps(
            ones,
            _mm_abs_ps(_mm_sub_ps(_mm_fmod_ps(h_prime, _mm_set1_ps(2f32)), ones)),
        ),
        c,
    );
    let zeros = _mm_setzero_ps();
    let m = _mm_sub_ps(v, c);
    let (mut r, mut g, mut b) = (zeros, zeros, zeros);
    let between_zero_and_one_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, zeros),
        _mm_cmplt_ps(h_prime, ones),
    );
    let twos = _mm_set1_ps(2f32);
    let between_one_and_two_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, ones),
        _mm_cmplt_ps(h_prime, twos),
    );
    let threes = _mm_set1_ps(3f32);
    let between_two_and_three_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, twos),
        _mm_cmplt_ps(h_prime, threes),
    );
    let fours = _mm_set1_ps(4f32);
    let between_three_and_four_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, threes),
        _mm_cmplt_ps(h_prime, fours),
    );
    let fives = _mm_set1_ps(5f32);
    let between_four_and_five_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, fours),
        _mm_cmplt_ps(h_prime, fives),
    );
    let between_five_and_six_mask = _mm_and_ps(
        _mm_cmpge_ps(h_prime, fives),
        _mm_cmplt_ps(h_prime, six),
    );
    // if h_prime >= 0f32 && h_prime < 1f32 {
    r = _mm_select_ps(between_zero_and_one_mask, c, r);
    g = _mm_select_ps(between_zero_and_one_mask, x, g);
    // if h_prime >= 1f32 && h_prime < 2f32 {
    r = _mm_select_ps(between_one_and_two_mask, x, r);
    g = _mm_select_ps(between_one_and_two_mask, c, g);
    // if h_prime >= 2f32 && h_prime < 3f32
    g = _mm_select_ps(between_two_and_three_mask, c, g);
    b = _mm_select_ps(between_two_and_three_mask, x, b);
    // if h_prime >= 3f32 && h_prime < 4f32 {
    g = _mm_select_ps(between_three_and_four_mask, x, g);
    b = _mm_select_ps(between_three_and_four_mask, c, b);
    // if h_prime >= 4f32 && h_prime < 5f32 {
    r = _mm_select_ps(between_four_and_five_mask, x, r);
    b = _mm_select_ps(between_four_and_five_mask, c, b);
    // if h_prime >= 5f32 && h_prime < 6f32 {
    r = _mm_select_ps(between_five_and_six_mask, c, r);
    b = _mm_select_ps(between_five_and_six_mask, x, b);
    r = _mm_add_ps(r, m);
    g = _mm_add_ps(g, m);
    b = _mm_add_ps(b, m);
    let rgb_scale = _mm_set1_ps(255f32);
    r = _mm_mul_ps(r, rgb_scale);
    g = _mm_mul_ps(g, rgb_scale);
    b = _mm_mul_ps(b, rgb_scale);
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    (
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(r)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(g)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(b)),
    )
}

#[inline(always)]
pub unsafe fn sse_rgb_to_hsv(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    scale: __m128,
) -> (__m128, __m128, __m128) {
    let rgb_scale = _mm_set1_ps(1f32 / 255f32);
    let r = _mm_mul_ps(_mm_cvtepi32_ps(r), rgb_scale);
    let g = _mm_mul_ps(_mm_cvtepi32_ps(g), rgb_scale);
    let b = _mm_mul_ps(_mm_cvtepi32_ps(b), rgb_scale);
    let c_max = _mm_max_ps(_mm_max_ps(r, g), b);
    let c_min = _mm_min_ps(_mm_min_ps(r, g), b);
    let delta = _mm_sub_ps(c_max, c_min);
    let rcp_delta = _mm_rcp_ps(delta);
    let is_r_max = _mm_cmpeq_ps(c_max, r);
    let is_g_max = _mm_cmpeq_ps(c_max, g);
    let is_b_max = _mm_cmpeq_ps(c_max, b);
    let immediate_zero_flag = _mm_cmpeq_ps(delta, _mm_setzero_ps());
    let mut h = _mm_setzero_ps();
    let v_six = _mm_set1_ps(60f32);
    h = _mm_select_ps(
        is_r_max,
        _mm_mul_ps(
            _mm_fmod_ps(_mm_mul_ps(_mm_sub_ps(g, b), rcp_delta), _mm_set1_ps(6f32)),
            v_six,
        ),
        h,
    );
    let adding_2 = _mm_set1_ps(2f32);
    h = _mm_select_ps(
        is_g_max,
        _mm_mul_ps(
            _mm_add_ps(_mm_mul_ps(_mm_sub_ps(b, r), rcp_delta), adding_2),
            v_six,
        ),
        h,
    );
    let adding_4 = _mm_set1_ps(4f32);
    h = _mm_select_ps(
        is_b_max,
        _mm_mul_ps(
            _mm_add_ps(_mm_mul_ps(_mm_sub_ps(r, g), rcp_delta), adding_4),
            v_six,
        ),
        h,
    );
    let zeros = _mm_setzero_ps();
    h = _mm_select_ps(immediate_zero_flag, zeros, h);
    let s = _mm_select_ps(
        _mm_cmpeq_ps(c_max, zeros),
        zeros,
        _mm_mul_ps(delta, _mm_rcp_ps(c_max)),
    );
    h = _mm_select_ps(
        _mm_cmplt_ps(h, zeros),
        _mm_add_ps(h, _mm_set1_ps(360f32)),
        h,
    );
    let v = c_max;
    (h, _mm_mul_ps(s, scale), _mm_mul_ps(v, scale))
}

#[inline(always)]
pub unsafe fn sse_rgb_to_hsl(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    scale: __m128,
) -> (__m128, __m128, __m128) {
    let rgb_scale = _mm_set1_ps(1f32 / 255f32);
    let r = _mm_mul_ps(_mm_cvtepi32_ps(r), rgb_scale);
    let g = _mm_mul_ps(_mm_cvtepi32_ps(g), rgb_scale);
    let b = _mm_mul_ps(_mm_cvtepi32_ps(b), rgb_scale);
    let c_max = _mm_max_ps(_mm_max_ps(r, g), b);
    let c_min = _mm_min_ps(_mm_min_ps(r, g), b);
    let delta = _mm_sub_ps(c_max, c_min);
    let rcp_delta = _mm_rcp_ps(delta);
    let is_r_max = _mm_cmpeq_ps(c_max, r);
    let is_g_max = _mm_cmpeq_ps(c_max, g);
    let is_b_max = _mm_cmpeq_ps(c_max, b);
    let zeros = _mm_setzero_ps();
    let immediate_zero_flag = _mm_cmpeq_ps(delta, zeros);
    let v_six = _mm_set1_ps(60f32);
    let mut h = _mm_setzero_ps();
    h = _mm_select_ps(
        is_r_max,
        _mm_mul_ps(
            _mm_fmod_ps(_mm_mul_ps(_mm_sub_ps(g, b), rcp_delta), _mm_set1_ps(6f32)),
            v_six,
        ),
        h,
    );
    let adding_2 = _mm_set1_ps(2f32);
    h = _mm_select_ps(
        is_g_max,
        _mm_mul_ps(
            _mm_add_ps(_mm_mul_ps(_mm_sub_ps(b, r), rcp_delta), adding_2),
            v_six,
        ),
        h,
    );
    let adding_4 = _mm_set1_ps(4f32);
    h = _mm_select_ps(
        is_b_max,
        _mm_mul_ps(
            _mm_add_ps(_mm_mul_ps(_mm_sub_ps(r, g), rcp_delta), adding_4),
            v_six,
        ),
        h,
    );
    h = _mm_select_ps(immediate_zero_flag, zeros, h);
    h = _mm_select_ps(
        _mm_cmplt_ps(h, zeros),
        _mm_add_ps(h, _mm_set1_ps(360f32)),
        h,
    );
    let l = _mm_mul_ps(_mm_add_ps(c_max, c_min), _mm_set1_ps(0.5f32));
    let s = _mm_div_ps(
        delta,
        _mm_sub_ps(
            _mm_set1_ps(1f32),
            _mm_abs_ps(_mm_prefer_fma_ps(_mm_set1_ps(-1f32), _mm_set1_ps(2f32), l)),
        ),
    );
    (h, _mm_mul_ps(s, scale), _mm_mul_ps(l, scale))
}
