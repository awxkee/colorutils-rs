#[allow(unused_imports)]
use crate::gamma_curves::TransferFunction;
use crate::neon::neon_math::vpowq_n_f32;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::*;

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[allow(dead_code)]
#[inline(always)]
pub unsafe fn neon_srgb_from_linear(linear: float32x4_t) -> float32x4_t {
    let low_cut_off = vdupq_n_f32(0.0030412825601275209f32);
    let mask = vcgeq_f32(linear, low_cut_off);

    let mut low = linear;
    let mut high = linear;
    low = vmulq_n_f32(low, 12.92f32);

    high = vsubq_f32(
        vmulq_n_f32(vpowq_n_f32(high, 1.0f32 / 2.4f32), 1.0550107189475866f32),
        vdupq_n_f32(0.0550107189475866f32),
    );
    return vbslq_f32(mask, high, low);
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[allow(dead_code)]
#[inline(always)]
pub unsafe fn neon_srgb_to_linear(gamma: float32x4_t) -> float32x4_t {
    let low_cut_off = vdupq_n_f32(12.92f32 * 0.0030412825601275209f32);
    let mask = vcgeq_f32(gamma, low_cut_off);

    let mut low = gamma;
    let high = vpowq_n_f32(
        vmulq_n_f32(
            vaddq_f32(gamma, vdupq_n_f32(0.0550107189475866f32)),
            1f32 / 1.0550107189475866f32,
        ),
        2.4f32,
    );
    low = vmulq_n_f32(low, 1f32 / 12.92f32);
    return vbslq_f32(mask, high, low);
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[allow(dead_code)]
#[inline(always)]
pub unsafe fn neon_rec709_from_linear(linear: float32x4_t) -> float32x4_t {
    let low_cut_off = vdupq_n_f32(0.018053968510807f32);
    let mask = vcgeq_f32(linear, low_cut_off);

    let mut low = linear;
    let mut high = linear;
    low = vmulq_n_f32(low, 4.5f32);

    high = vsubq_f32(
        vmulq_n_f32(vpowq_n_f32(high, 0.45f32), 1.09929682680944f32),
        vdupq_n_f32(0.09929682680944f32),
    );
    return vbslq_f32(mask, high, low);
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[allow(dead_code)]
#[inline(always)]
pub unsafe fn neon_rec709_to_linear(linear: float32x4_t) -> float32x4_t {
    let low_cut_off = vdupq_n_f32(4.5f32 * 0.018053968510807f32);
    let mask = vcgeq_f32(linear, low_cut_off);

    let mut low = linear;
    let high = vpowq_n_f32(
        vmulq_n_f32(
            vaddq_f32(linear, vdupq_n_f32(0.09929682680944f32)),
            1f32 / 1.09929682680944f32,
        ),
        1.0f32 / 0.45f32,
    );
    low = vmulq_n_f32(low, 1f32 / 4.5f32);
    return vbslq_f32(mask, high, low);
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
pub unsafe fn get_neon_linear_transfer(
    transfer_function: TransferFunction,
) -> unsafe fn(float32x4_t) -> float32x4_t {
    match transfer_function {
        TransferFunction::Srgb => neon_srgb_to_linear,
        TransferFunction::Rec709 => neon_rec709_to_linear,
    }
}
