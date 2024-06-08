use crate::neon::neon_math::{prefer_vfmaq_f32, vfmodq_f32};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_hsl_to_rgb(
    h: float32x4_t,
    s: float32x4_t,
    l: float32x4_t,
    scale: float32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t) {
    let s = vmulq_f32(s, scale);
    let l = vmulq_f32(l, scale);
    let ones = vdupq_n_f32(1f32);
    let c = vmulq_f32(
        vsubq_f32(ones, vabsq_f32(vsubq_f32(vmulq_n_f32(l, 2f32), ones))),
        s,
    );
    let x = vmulq_f32(vsubq_f32(
        ones,
        vabsq_f32(vsubq_f32(
            vfmodq_f32(vmulq_n_f32(h, 1f32 / 60f32), vdupq_n_f32(2f32)),
            ones,
        )),
    ), c);

    let zeros = vdupq_n_f32(0f32);
    let m = vsubq_f32(l, vmulq_n_f32(c, 0.5f32));
    let h_prime = h;
    let (mut r, mut g, mut b) = (zeros, zeros, zeros);
    let between_zero_and_one_mask =
        vandq_u32(vcgezq_f32(h), vcltq_f32(h_prime, vdupq_n_f32(60f32)));
    let between_one_and_two_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(60f32)),
        vcltq_f32(h_prime, vdupq_n_f32(120f32)),
    );
    let between_two_and_three_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(120f32)),
        vcltq_f32(h_prime, vdupq_n_f32(180f32)),
    );
    let between_three_and_four_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(180f32)),
        vcltq_f32(h_prime, vdupq_n_f32(240f32)),
    );
    let between_four_and_five_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(240f32)),
        vcltq_f32(h_prime, vdupq_n_f32(300f32)),
    );
    let between_five_and_six_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(300f32)),
        vcltq_f32(h_prime, vdupq_n_f32(360f32)),
    );
    // if h_prime >= 0f32 && h_prime < 1f32 {
    r = vbslq_f32(between_zero_and_one_mask, c, r);
    g = vbslq_f32(between_zero_and_one_mask, x, g);
    // if h_prime >= 1f32 && h_prime < 2f32 {
    r = vbslq_f32(between_one_and_two_mask, x, r);
    g = vbslq_f32(between_one_and_two_mask, c, g);
    // if h_prime >= 2f32 && h_prime < 3f32
    g = vbslq_f32(between_two_and_three_mask, c, g);
    b = vbslq_f32(between_two_and_three_mask, x, b);
    // if h_prime >= 3f32 && h_prime < 4f32 {
    g = vbslq_f32(between_three_and_four_mask, x, g);
    b = vbslq_f32(between_three_and_four_mask, c, b);
    // if h_prime >= 4f32 && h_prime < 5f32 {
    r = vbslq_f32(between_four_and_five_mask, x, r);
    b = vbslq_f32(between_four_and_five_mask, c, b);
    // if h_prime >= 5f32 && h_prime < 6f32 {
    r = vbslq_f32(between_five_and_six_mask, c, r);
    b = vbslq_f32(between_five_and_six_mask, x, b);
    r = vaddq_f32(r, m);
    g = vaddq_f32(g, m);
    b = vaddq_f32(b, m);
    r = vmulq_n_f32(r, 255f32);
    g = vmulq_n_f32(g, 255f32);
    b = vmulq_n_f32(b, 255f32);
    (vcvtaq_u32_f32(r), vcvtaq_u32_f32(g), vcvtaq_u32_f32(b))
}

#[inline(always)]
pub unsafe fn neon_hsv_to_rgb(
    h: float32x4_t,
    s: float32x4_t,
    v: float32x4_t,
    scale: float32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t) {
    let s = vmulq_f32(s, scale);
    let v = vmulq_f32(v, scale);
    let c = vmulq_f32(s, v);
    let h_prime = vfmodq_f32(vmulq_n_f32(h, 1f32 / 60f32), vdupq_n_f32(6f32));
    let ones = vdupq_n_f32(1f32);
    let x = vmulq_f32(
        vsubq_f32(
            ones,
            vabsq_f32(vsubq_f32(vfmodq_f32(h_prime, vdupq_n_f32(2f32)), ones)),
        ),
        c,
    );
    let zeros = vdupq_n_f32(0f32);
    let m = vsubq_f32(v, c);
    let (mut r, mut g, mut b) = (zeros, zeros, zeros);
    let between_zero_and_one_mask =
        vandq_u32(vcgezq_f32(h_prime), vcltq_f32(h_prime, vdupq_n_f32(1f32)));
    let between_one_and_two_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(1f32)),
        vcltq_f32(h_prime, vdupq_n_f32(2f32)),
    );
    let between_two_and_three_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(2f32)),
        vcltq_f32(h_prime, vdupq_n_f32(3f32)),
    );
    let between_three_and_four_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(3f32)),
        vcltq_f32(h_prime, vdupq_n_f32(4f32)),
    );
    let between_four_and_five_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(4f32)),
        vcltq_f32(h_prime, vdupq_n_f32(5f32)),
    );
    let between_five_and_six_mask = vandq_u32(
        vcgeq_f32(h_prime, vdupq_n_f32(5f32)),
        vcltq_f32(h_prime, vdupq_n_f32(6f32)),
    );
    // if h_prime >= 0f32 && h_prime < 1f32 {
    r = vbslq_f32(between_zero_and_one_mask, c, r);
    g = vbslq_f32(between_zero_and_one_mask, x, g);
    // if h_prime >= 1f32 && h_prime < 2f32 {
    r = vbslq_f32(between_one_and_two_mask, x, r);
    g = vbslq_f32(between_one_and_two_mask, c, g);
    // if h_prime >= 2f32 && h_prime < 3f32
    g = vbslq_f32(between_two_and_three_mask, c, g);
    b = vbslq_f32(between_two_and_three_mask, x, b);
    // if h_prime >= 3f32 && h_prime < 4f32 {
    g = vbslq_f32(between_three_and_four_mask, x, g);
    b = vbslq_f32(between_three_and_four_mask, c, b);
    // if h_prime >= 4f32 && h_prime < 5f32 {
    r = vbslq_f32(between_four_and_five_mask, x, r);
    b = vbslq_f32(between_four_and_five_mask, c, b);
    // if h_prime >= 5f32 && h_prime < 6f32 {
    r = vbslq_f32(between_five_and_six_mask, c, r);
    b = vbslq_f32(between_five_and_six_mask, x, b);
    r = vaddq_f32(r, m);
    g = vaddq_f32(g, m);
    b = vaddq_f32(b, m);
    r = vmulq_n_f32(r, 255f32);
    g = vmulq_n_f32(g, 255f32);
    b = vmulq_n_f32(b, 255f32);
    (vcvtaq_u32_f32(r), vcvtaq_u32_f32(g), vcvtaq_u32_f32(b))
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
pub unsafe fn neon_rgb_to_hsv(
    r: uint32x4_t,
    g: uint32x4_t,
    b: uint32x4_t,
    scale: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let rgb_scale = vdupq_n_f32(1f32 / 255f32);
    let r = vmulq_f32(vcvtq_f32_u32(r), rgb_scale);
    let g = vmulq_f32(vcvtq_f32_u32(g), rgb_scale);
    let b = vmulq_f32(vcvtq_f32_u32(b), rgb_scale);
    let c_max = vmaxq_f32(vmaxq_f32(r, g), b);
    let c_min = vminq_f32(vminq_f32(r, g), b);
    let delta = vsubq_f32(c_max, c_min);
    let rcp_delta = vrecpeq_f32(delta);
    let is_r_max = vceqq_f32(c_max, r);
    let is_g_max = vceqq_f32(c_max, g);
    let is_b_max = vceqq_f32(c_max, b);
    let immediate_zero_flag = vceqzq_f32(delta);
    let mut h = vdupq_n_f32(0f32);
    h = vbslq_f32(
        is_r_max,
        vmulq_n_f32(
            vfmodq_f32(vmulq_f32(vsubq_f32(g, b), rcp_delta), vdupq_n_f32(6f32)),
            60f32,
        ),
        h,
    );
    let adding_2 = vdupq_n_f32(2f32);
    h = vbslq_f32(
        is_g_max,
        vmulq_n_f32(
            vaddq_f32(vmulq_f32(vsubq_f32(b, r), rcp_delta), adding_2),
            60f32,
        ),
        h,
    );
    let adding_4 = vdupq_n_f32(4f32);
    h = vbslq_f32(
        is_b_max,
        vmulq_n_f32(
            vaddq_f32(vmulq_f32(vsubq_f32(r, g), rcp_delta), adding_4),
            60f32,
        ),
        h,
    );
    let zeros = vdupq_n_f32(0f32);
    h = vbslq_f32(immediate_zero_flag, zeros, h);
    let s = vbslq_f32(
        vceqzq_f32(c_max),
        zeros,
        vmulq_f32(delta, vrecpeq_f32(c_max)),
    );
    h = vbslq_f32(vcltzq_f32(h), vaddq_f32(h, vdupq_n_f32(360f32)), h);
    let v = c_max;
    (h, vmulq_f32(s, scale), vmulq_f32(v, scale))
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
pub unsafe fn neon_rgb_to_hsl(
    r: uint32x4_t,
    g: uint32x4_t,
    b: uint32x4_t,
    scale: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let rgb_scale = vdupq_n_f32(1f32 / 255f32);
    let r = vmulq_f32(vcvtq_f32_u32(r), rgb_scale);
    let g = vmulq_f32(vcvtq_f32_u32(g), rgb_scale);
    let b = vmulq_f32(vcvtq_f32_u32(b), rgb_scale);
    let c_max = vmaxq_f32(vmaxq_f32(r, g), b);
    let c_min = vminq_f32(vminq_f32(r, g), b);
    let delta = vsubq_f32(c_max, c_min);
    let rcp_delta = vrecpeq_f32(delta);
    let is_r_max = vceqq_f32(c_max, r);
    let is_g_max = vceqq_f32(c_max, g);
    let is_b_max = vceqq_f32(c_max, b);
    let immediate_zero_flag = vceqzq_f32(delta);
    let mut h = vdupq_n_f32(0f32);
    h = vbslq_f32(
        is_r_max,
        vmulq_n_f32(
            vfmodq_f32(vmulq_f32(vsubq_f32(g, b), rcp_delta), vdupq_n_f32(6f32)),
            60f32,
        ),
        h,
    );
    let adding_2 = vdupq_n_f32(2f32);
    h = vbslq_f32(
        is_g_max,
        vmulq_n_f32(
            vaddq_f32(vmulq_f32(vsubq_f32(b, r), rcp_delta), adding_2),
            60f32,
        ),
        h,
    );
    let adding_4 = vdupq_n_f32(4f32);
    h = vbslq_f32(
        is_b_max,
        vmulq_n_f32(
            vaddq_f32(vmulq_f32(vsubq_f32(r, g), rcp_delta), adding_4),
            60f32,
        ),
        h,
    );
    let zeros = vdupq_n_f32(0f32);
    h = vbslq_f32(immediate_zero_flag, zeros, h);
    h = vbslq_f32(vcltzq_f32(h), vaddq_f32(h, vdupq_n_f32(360f32)), h);
    let l = vmulq_n_f32(vaddq_f32(c_max, c_min), 0.5f32);
    let s = vdivq_f32(
        delta,
        vsubq_f32(
            vdupq_n_f32(1f32),
            vabsq_f32(prefer_vfmaq_f32(vdupq_n_f32(-1f32), vdupq_n_f32(2f32), l)),
        ),
    );
    (h, vmulq_f32(s, scale), vmulq_f32(l, scale))
}
