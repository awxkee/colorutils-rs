#[allow(unused_imports)]
use crate::image::ImageConfiguration;
#[allow(unused_imports)]
use crate::neon::*;
#[allow(unused_imports)]
use crate::TransferFunction;
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
pub unsafe fn get_neon_gamma_transfer(
    transfer_function: TransferFunction,
) -> unsafe fn(float32x4_t) -> float32x4_t {
    match transfer_function {
        TransferFunction::Srgb => neon_srgb_from_linear,
        TransferFunction::Rec709 => neon_rec709_from_linear,
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
unsafe fn neon_gamma_vld<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: *const f32,
    transfer_function: TransferFunction,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
    let d_alpha = vdupq_n_f32(1f32);
    let transfer = get_neon_gamma_transfer(transfer_function);
    let v_scale_alpha = vdupq_n_f32(255f32);
    let (mut r_f32, mut g_f32, mut b_f32, mut a_f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    match image_configuration {
        ImageConfiguration::Rgba | ImageConfiguration::Bgra => {
            let rgba_pixels = vld4q_f32(src);
            if image_configuration == ImageConfiguration::Rgba {
                r_f32 = rgba_pixels.0;
                g_f32 = rgba_pixels.1;
                b_f32 = rgba_pixels.2;
            } else {
                r_f32 = rgba_pixels.2;
                g_f32 = rgba_pixels.1;
                b_f32 = rgba_pixels.0;
            }
            a_f32 = rgba_pixels.3;
        }
        ImageConfiguration::Bgr | ImageConfiguration::Rgb => {
            let rgb_pixels = vld3q_f32(src);
            if image_configuration == ImageConfiguration::Rgb {
                r_f32 = rgb_pixels.0;
                g_f32 = rgb_pixels.1;
                b_f32 = rgb_pixels.2;
            } else {
                r_f32 = rgb_pixels.2;
                g_f32 = rgb_pixels.1;
                b_f32 = rgb_pixels.0;
            }
            a_f32 = d_alpha;
        }
    }

    r_f32 = transfer(r_f32);
    g_f32 = transfer(g_f32);
    b_f32 = transfer(b_f32);
    r_f32 = vmulq_f32(r_f32, v_scale_alpha);
    g_f32 = vmulq_f32(g_f32, v_scale_alpha);
    b_f32 = vmulq_f32(b_f32, v_scale_alpha);
    if USE_ALPHA {
        a_f32 = vmulq_f32(a_f32, v_scale_alpha);
    }
    (
        vcvtaq_u32_f32(r_f32),
        vcvtaq_u32_f32(g_f32),
        vcvtaq_u32_f32(b_f32),
        vcvtaq_u32_f32(a_f32),
    )
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
#[inline(always)]
pub unsafe fn neon_linear_to_gamma<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_0, transfer_function);

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_1, transfer_function);

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_2, transfer_function);

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            neon_gamma_vld::<CHANNELS_CONFIGURATION, USE_ALPHA>(src_ptr_3, transfer_function);

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), vqmovn_u32(r_row1_));
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), vqmovn_u32(g_row1_));
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), vqmovn_u32(b_row1_));

        let r_row23 = vcombine_u16(vqmovn_u32(r_row2_), vqmovn_u32(r_row3_));
        let g_row23 = vcombine_u16(vqmovn_u32(g_row2_), vqmovn_u32(g_row3_));
        let b_row23 = vcombine_u16(vqmovn_u32(b_row2_), vqmovn_u32(b_row3_));

        let r_row = vcombine_u8(vqmovn_u16(r_row01), vqmovn_u16(r_row23));
        let g_row = vcombine_u8(vqmovn_u16(g_row01), vqmovn_u16(g_row23));
        let b_row = vcombine_u8(vqmovn_u16(b_row01), vqmovn_u16(b_row23));

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if USE_ALPHA {
            let a_row01 = vcombine_u16(vqmovn_u32(a_row0_), vqmovn_u32(a_row1_));
            let a_row23 = vcombine_u16(vqmovn_u32(a_row2_), vqmovn_u32(a_row3_));
            let a_row = vcombine_u8(vqmovn_u16(a_row01), vqmovn_u16(a_row23));
            let store_rows = uint8x16x4_t(r_row, g_row, b_row, a_row);
            vst4q_u8(dst_ptr, store_rows);
        } else {
            let store_rows = uint8x16x3_t(r_row, g_row, b_row);
            vst3q_u8(dst_ptr, store_rows);
        }

        cx += 16;
    }

    cx
}
