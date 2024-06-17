use crate::sse::{_mm_abs_ps, _mm_fmod_ps, _mm_prefer_fma_ps, _mm_select_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
    let between_zero_and_one_mask =
        _mm_and_ps(_mm_cmpge_ps(h_prime, zeros), _mm_cmplt_ps(h_prime, ones));
    let twos = _mm_set1_ps(2f32);
    let between_one_and_two_mask =
        _mm_and_ps(_mm_cmpge_ps(h_prime, ones), _mm_cmplt_ps(h_prime, twos));
    let threes = _mm_set1_ps(3f32);
    let between_two_and_three_mask =
        _mm_and_ps(_mm_cmpge_ps(h_prime, twos), _mm_cmplt_ps(h_prime, threes));
    let fours = _mm_set1_ps(4f32);
    let between_three_and_four_mask =
        _mm_and_ps(_mm_cmpge_ps(h_prime, threes), _mm_cmplt_ps(h_prime, fours));
    let fives = _mm_set1_ps(5f32);
    let between_four_and_five_mask =
        _mm_and_ps(_mm_cmpge_ps(h_prime, fours), _mm_cmplt_ps(h_prime, fives));
    let between_five_and_six_mask =
        _mm_and_ps(_mm_cmpge_ps(h_prime, fives), _mm_cmplt_ps(h_prime, six));
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
