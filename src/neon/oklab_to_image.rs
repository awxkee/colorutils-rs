/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::neon::get_neon_gamma_transfer;
use crate::neon::math::vcolorq_matrix_f32;
use crate::TransferFunction;
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_oklab_gamma_vld<const CHANNELS_CONFIGURATION: u8>(
    src: *const f32,
    transfer_function: TransferFunction,
    m0: float32x4_t,
    m1: float32x4_t,
    m2: float32x4_t,
    m3: float32x4_t,
    m4: float32x4_t,
    m5: float32x4_t,
    m6: float32x4_t,
    m7: float32x4_t,
    m8: float32x4_t,
    c0: float32x4_t,
    c1: float32x4_t,
    c2: float32x4_t,
    c3: float32x4_t,
    c4: float32x4_t,
    c5: float32x4_t,
    c6: float32x4_t,
    c7: float32x4_t,
    c8: float32x4_t,
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

    let (mut l_l, mut l_m, mut l_s) =
        vcolorq_matrix_f32(r_f32, g_f32, b_f32, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    l_l = vmulq_f32(vmulq_f32(l_l, l_l), l_l);
    l_m = vmulq_f32(vmulq_f32(l_m, l_m), l_m);
    l_s = vmulq_f32(vmulq_f32(l_s, l_s), l_s);

    let (r_l, g_l, b_l) = vcolorq_matrix_f32(l_l, l_m, l_s, c0, c1, c2, c3, c4, c5, c6, c7, c8);

    r_f32 = transfer(r_l);
    g_f32 = transfer(g_l);
    b_f32 = transfer(b_l);
    r_f32 = vmulq_f32(r_f32, v_scale_alpha);
    g_f32 = vmulq_f32(g_f32, v_scale_alpha);
    b_f32 = vmulq_f32(b_f32, v_scale_alpha);
    if image_configuration.has_alpha() {
        a_f32 = vmulq_f32(a_f32, v_scale_alpha);
    }
    (
        vcvtaq_u32_f32(r_f32),
        vcvtaq_u32_f32(g_f32),
        vcvtaq_u32_f32(b_f32),
        vcvtaq_u32_f32(a_f32),
    )
}

#[inline(always)]
pub unsafe fn neon_oklab_to_image<const CHANNELS_CONFIGURATION: u8>(
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

    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        vdupq_n_f32(1f32),
        vdupq_n_f32(0.3963377774f32),
        vdupq_n_f32(0.2158037573f32),
        vdupq_n_f32(1f32),
        vdupq_n_f32(-0.1055613458f32),
        vdupq_n_f32(-0.0638541728f32),
        vdupq_n_f32(1f32),
        vdupq_n_f32(-0.0894841775f32),
        vdupq_n_f32(-1.2914855480f32),
    );

    let (c0, c1, c2, c3, c4, c5, c6, c7, c8) = (
        vdupq_n_f32(4.0767416621f32),
        vdupq_n_f32(-3.3077115913f32),
        vdupq_n_f32(0.2309699292f32),
        vdupq_n_f32(-1.2684380046f32),
        vdupq_n_f32(2.6097574011f32),
        vdupq_n_f32(-0.3413193965f32),
        vdupq_n_f32(-0.0041960863f32),
        vdupq_n_f32(-0.7034186147f32),
        vdupq_n_f32(1.7076147010f32),
    );

    while cx + 16 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) = neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_0,
            transfer_function,
            m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
        );

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) = neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_1,
            transfer_function,
            m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
        );

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) = neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_2,
            transfer_function,
            m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
        );

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) = neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION>(
            src_ptr_3,
            transfer_function,
            m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
        );

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

        if image_configuration.has_alpha() {
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