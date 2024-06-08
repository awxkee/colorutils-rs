#[allow(unused_imports)]
use crate::gamma_curves::TransferFunction;
#[allow(unused_imports)]
use crate::image::ImageConfiguration;
#[allow(unused_imports)]
use crate::image_to_xyz_lab::XyzTarget;
#[allow(unused_imports)]
use crate::luv::{LUV_CUTOFF_FORWARD_Y, LUV_MULTIPLIER_FORWARD_Y};
use crate::neon::neon_math::*;
#[allow(unused_imports)]
use crate::neon::*;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::*;

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
pub(crate) unsafe fn neon_triple_to_xyz(
    r: uint32x4_t,
    g: uint32x4_t,
    b: uint32x4_t,
    c1: float32x4_t,
    c2: float32x4_t,
    c3: float32x4_t,
    c4: float32x4_t,
    c5: float32x4_t,
    c6: float32x4_t,
    c7: float32x4_t,
    c8: float32x4_t,
    c9: float32x4_t,
    transfer: &unsafe fn(float32x4_t) -> float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let r_f = vmulq_n_f32(vcvtq_f32_u32(r), 1f32 / 255f32);
    let g_f = vmulq_n_f32(vcvtq_f32_u32(g), 1f32 / 255f32);
    let b_f = vmulq_n_f32(vcvtq_f32_u32(b), 1f32 / 255f32);
    let r_linear = transfer(r_f);
    let g_linear = transfer(g_f);
    let b_linear = transfer(b_f);

    let (x, y, z) = vcolorq_matrix_f32(
        r_linear, g_linear, b_linear, c1, c2, c3, c4, c5, c6, c7, c8, c9,
    );
    (x, y, z)
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
pub(crate) unsafe fn neon_triple_to_luv(
    x: float32x4_t,
    y: float32x4_t,
    z: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let zeros = vdupq_n_f32(0f32);
    let den = prefer_vfmaq_f32(
        prefer_vfmaq_f32(x, z, vdupq_n_f32(3f32)),
        y,
        vdupq_n_f32(15f32),
    );
    let nan_mask = vceqzq_f32(den);
    let l_low_mask = vcltq_f32(y, vdupq_n_f32(LUV_CUTOFF_FORWARD_Y));
    let y_cbrt = vcbrtq_f32(y);
    let l = vbslq_f32(
        l_low_mask,
        vmulq_n_f32(y, LUV_MULTIPLIER_FORWARD_Y),
        prefer_vfmaq_f32(vdupq_n_f32(-16f32), y_cbrt, vdupq_n_f32(116f32)),
    );
    let u_prime = vdivq_f32(vmulq_n_f32(x, 4f32), den);
    let v_prime = vdivq_f32(vmulq_n_f32(y, 9f32), den);
    let sub_u_prime = vsubq_f32(u_prime, vdupq_n_f32(crate::luv::LUV_WHITE_U_PRIME));
    let sub_v_prime = vsubq_f32(v_prime, vdupq_n_f32(crate::luv::LUV_WHITE_V_PRIME));
    let l13 = vmulq_n_f32(l, 13f32);
    let u = vbslq_f32(nan_mask, zeros, vmulq_f32(l13, sub_u_prime));
    let v = vbslq_f32(nan_mask, zeros, vmulq_f32(l13, sub_v_prime));
    (l, u, v)
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
pub(crate) unsafe fn neon_triple_to_lab(
    x: float32x4_t,
    y: float32x4_t,
    z: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let x = vmulq_n_f32(x, 100f32 / 95.047f32);
    let z = vmulq_n_f32(z, 100f32 / 108.883f32);
    let cbrt_x = vcbrtq_f32(x);
    let cbrt_y = vcbrtq_f32(y);
    let cbrt_z = vcbrtq_f32(z);
    let s_1 = vdupq_n_f32(16f32 / 116f32);
    let s_2 = vdupq_n_f32(7.787f32);
    let lower_x = prefer_vfmaq_f32(s_1, s_2, x);
    let lower_y = prefer_vfmaq_f32(s_1, s_2, y);
    let lower_z = prefer_vfmaq_f32(s_1, s_2, z);
    let kappa = vdupq_n_f32(0.008856f32);
    let x = vbslq_f32(vcgtq_f32(x, kappa), cbrt_x, lower_x);
    let y = vbslq_f32(vcgtq_f32(y, kappa), cbrt_y, lower_y);
    let z = vbslq_f32(vcgtq_f32(z, kappa), cbrt_z, lower_z);
    let l = prefer_vfmaq_f32(vdupq_n_f32(-16.0f32), y, vdupq_n_f32(116.0f32));
    let a = vmulq_n_f32(vsubq_f32(x, y), 500f32);
    let b = vmulq_n_f32(vsubq_f32(y, z), 200f32);
    (l, a, b)
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
pub unsafe fn neon_channels_to_xyz_or_lab<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    a_linearized: *mut f32,
    a_offset: usize,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> usize {
    if USE_ALPHA {
        if a_linearized.is_null() {
            panic!("Null alpha channel with requirements of linearized alpha if not supported");
        }
    }
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer = get_neon_linear_transfer(transfer_function);

    let cq1 = vdupq_n_f32(matrix[0][0]);
    let cq2 = vdupq_n_f32(matrix[0][1]);
    let cq3 = vdupq_n_f32(matrix[0][2]);
    let cq4 = vdupq_n_f32(matrix[1][0]);
    let cq5 = vdupq_n_f32(matrix[1][1]);
    let cq6 = vdupq_n_f32(matrix[1][2]);
    let cq7 = vdupq_n_f32(matrix[2][0]);
    let cq8 = vdupq_n_f32(matrix[2][1]);
    let cq9 = vdupq_n_f32(matrix[2][2]);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 16 < width as usize {
        let (r_chan, g_chan, b_chan, a_chan);
        let src_ptr = src.add(src_offset + cx * channels);
        match image_configuration {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => {
                let ldr = vld3q_u8(src_ptr);
                if image_configuration == ImageConfiguration::Rgb {
                    r_chan = ldr.0;
                    g_chan = ldr.1;
                    b_chan = ldr.2;
                } else {
                    r_chan = ldr.2;
                    g_chan = ldr.1;
                    b_chan = ldr.0;
                }
                a_chan = vdupq_n_u8(0);
            }
            ImageConfiguration::Rgba => {
                let ldr = vld4q_u8(src_ptr);
                r_chan = ldr.0;
                g_chan = ldr.1;
                b_chan = ldr.2;
                a_chan = ldr.3;
            }
            ImageConfiguration::Bgra => {
                let ldr = vld4q_u8(src_ptr);
                r_chan = ldr.2;
                g_chan = ldr.1;
                b_chan = ldr.0;
                a_chan = ldr.3;
            }
        }

        let r_low = vmovl_u8(vget_low_u8(r_chan));
        let g_low = vmovl_u8(vget_low_u8(g_chan));
        let b_low = vmovl_u8(vget_low_u8(b_chan));

        let r_low_low = vmovl_u16(vget_low_u16(r_low));
        let g_low_low = vmovl_u16(vget_low_u16(g_low));
        let b_low_low = vmovl_u16(vget_low_u16(b_low));

        let (mut x_low_low, mut y_low_low, mut z_low_low) = neon_triple_to_xyz(
            r_low_low, g_low_low, b_low_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9, &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = neon_triple_to_lab(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = a;
                z_low_low = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = neon_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
        }

        let xyz_low_low = float32x4x3_t(x_low_low, y_low_low, z_low_low);
        vst3q_f32(dst_ptr.add(cx * 3), xyz_low_low);

        let r_low_high = vmovl_high_u16(r_low);
        let g_low_high = vmovl_high_u16(g_low);
        let b_low_high = vmovl_high_u16(b_low);

        let (mut x_low_high, mut y_low_high, mut z_low_high) = neon_triple_to_xyz(
            r_low_high, g_low_high, b_low_high, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = neon_triple_to_lab(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = a;
                z_low_high = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = neon_triple_to_luv(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = u;
                z_low_high = v;
            }
        }

        let xyz_low_low = float32x4x3_t(x_low_high, y_low_high, z_low_high);
        vst3q_f32(dst_ptr.add(cx * 3 + 4 * 3), xyz_low_low);

        let r_high = vmovl_high_u8(r_chan);
        let g_high = vmovl_high_u8(g_chan);
        let b_high = vmovl_high_u8(b_chan);

        let r_high_low = vmovl_u16(vget_low_u16(r_high));
        let g_high_low = vmovl_u16(vget_low_u16(g_high));
        let b_high_low = vmovl_u16(vget_low_u16(b_high));

        let (mut x_high_low, mut y_high_low, mut z_high_low) = neon_triple_to_xyz(
            r_high_low, g_high_low, b_high_low, cq1, cq2, cq3, cq4, cq5, cq6, cq7, cq8, cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = neon_triple_to_lab(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = a;
                z_high_low = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = neon_triple_to_luv(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = u;
                z_high_low = v;
            }
        }

        let xyz_low_low = float32x4x3_t(x_high_low, y_high_low, z_high_low);
        vst3q_f32(dst_ptr.add(cx * 3 + 4 * 3 * 2), xyz_low_low);

        let r_high_high = vmovl_high_u16(r_high);
        let g_high_high = vmovl_high_u16(g_high);
        let b_high_high = vmovl_high_u16(b_high);

        let (mut x_high_high, mut y_high_high, mut z_high_high) = neon_triple_to_xyz(
            r_high_high,
            g_high_high,
            b_high_high,
            cq1,
            cq2,
            cq3,
            cq4,
            cq5,
            cq6,
            cq7,
            cq8,
            cq9,
            &transfer,
        );

        match target {
            XyzTarget::LAB => {
                let (l, a, b) = neon_triple_to_lab(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = a;
                z_high_high = b;
            }
            XyzTarget::XYZ => {}
            XyzTarget::LUV => {
                let (l, u, v) = neon_triple_to_luv(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = u;
                z_high_high = v;
            }
        }

        let xyz_low_low = float32x4x3_t(x_high_high, y_high_high, z_high_high);
        vst3q_f32(dst_ptr.add(cx * 3 + 4 * 3 * 3), xyz_low_low);

        if USE_ALPHA {
            let a_ptr = (a_linearized as *mut u8).add(a_offset) as *mut f32;

            let a_low = vmovl_u8(vget_low_u8(a_chan));

            let a_low_low =
                vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_low))), 1f32 / 255f32);

            vst1q_f32(a_ptr.add(cx), a_low_low);

            let a_low_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_low)), 1f32 / 255f32);

            vst1q_f32(a_ptr.add(cx + 4), a_low_high);

            let a_high = vmovl_high_u8(a_chan);

            let a_high_low = vmulq_n_f32(
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_high))),
                1f32 / 255f32,
            );

            vst1q_f32(a_ptr.add(cx + 4 * 2), a_high_low);

            let a_high_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_high)), 1f32 / 255f32);

            vst1q_f32(a_ptr.add(cx + 4 * 3), a_high_high);
        }

        cx += 16;
    }

    cx
}
