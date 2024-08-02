/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use std::arch::aarch64::*;

use erydanos::{vcosq_f32, vsinq_f32};

use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
use crate::neon::get_neon_gamma_transfer;
use crate::neon::math::vcolorq_matrix_f32;
use crate::{load_f32_and_deinterleave_direct, TransferFunction, XYZ_TO_SRGB_D65};

#[inline(always)]
unsafe fn neon_oklab_gamma_vld<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: *const f32,
    transfer: &unsafe fn(float32x4_t) -> float32x4_t,
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
    x0: float32x4_t,
    x1: float32x4_t,
    x2: float32x4_t,
    x3: float32x4_t,
    x4: float32x4_t,
    x5: float32x4_t,
    x6: float32x4_t,
    x7: float32x4_t,
    x8: float32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
    let target: OklabTarget = TARGET.into();
    let v_scale_alpha = vdupq_n_f32(255f32);
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let (l, mut a, mut b, mut a_f32) = load_f32_and_deinterleave_direct!(src, image_configuration);

    if target == OklabTarget::OKLCH {
        let a0 = vmulq_f32(a, vcosq_f32(b));
        let b0 = vmulq_f32(a, vsinq_f32(b));
        a = a0;
        b = b0;
    }

    let (mut l_l, mut l_m, mut l_s) =
        vcolorq_matrix_f32(l, a, b, m0, m1, m2, m3, m4, m5, m6, m7, m8);

    l_l = vmulq_f32(vmulq_f32(l_l, l_l), l_l);
    l_m = vmulq_f32(vmulq_f32(l_m, l_m), l_m);
    l_s = vmulq_f32(vmulq_f32(l_s, l_s), l_s);

    let (x, y, z) = vcolorq_matrix_f32(l_l, l_m, l_s, c0, c1, c2, c3, c4, c5, c6, c7, c8);

    let (r_l, g_l, b_l) = vcolorq_matrix_f32(x, y, z, x0, x1, x2, x3, x4, x5, x6, x7, x8);

    let mut r_f32 = transfer(r_l);
    let mut g_f32 = transfer(g_l);
    let mut b_f32 = transfer(b_l);
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
pub unsafe fn neon_oklab_to_image<
    const CHANNELS_CONFIGURATION: u8,
    const TARGET: u8,
    const TRANSFER_FUNCTION: u8,
>(
    start_cx: usize,
    src: *const f32,
    src_offset: u32,
    dst: *mut u8,
    dst_offset: u32,
    width: u32,
    _: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer_function: TransferFunction = TRANSFER_FUNCTION.into();

    let transfer = get_neon_gamma_transfer(transfer_function);

    // Matrix from XYZ
    let (x0, x1, x2, x3, x4, x5, x6, x7, x8) = (
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(0)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(1)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(0).get_unchecked(2)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(0)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(1)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(1).get_unchecked(2)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(0)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(1)),
        vdupq_n_f32(*XYZ_TO_SRGB_D65.get_unchecked(2).get_unchecked(2)),
    );

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

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION, TARGET>(
                src_ptr_0, &transfer, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4, c5,
                c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            );

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION, TARGET>(
                src_ptr_1, &transfer, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4, c5,
                c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            );

        let src_ptr_2 = offset_src_ptr.add(4 * 2 * channels);

        let (r_row2_, g_row2_, b_row2_, a_row2_) =
            neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION, TARGET>(
                src_ptr_2, &transfer, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4, c5,
                c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            );

        let src_ptr_3 = offset_src_ptr.add(4 * 3 * channels);

        let (r_row3_, g_row3_, b_row3_, a_row3_) =
            neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION, TARGET>(
                src_ptr_3, &transfer, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4, c5,
                c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
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
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x16x4_t(r_row, g_row, b_row, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x16x4_t(b_row, g_row, r_row, a_row)
                }
            };
            vst4q_u8(dst_ptr, store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x16x3_t(r_row, g_row, b_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x16x3_t(b_row, g_row, r_row)
                }
            };
            vst3q_u8(dst_ptr, store_rows);
        }

        cx += 16;
    }

    while cx + 8 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION, TARGET>(
                src_ptr_0, &transfer, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4, c5,
                c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            );

        let src_ptr_1 = offset_src_ptr.add(4 * channels);

        let (r_row1_, g_row1_, b_row1_, a_row1_) =
            neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION, TARGET>(
                src_ptr_1, &transfer, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4, c5,
                c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            );

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), vqmovn_u32(r_row1_));
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), vqmovn_u32(g_row1_));
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), vqmovn_u32(b_row1_));

        let r_row = vqmovn_u16(r_row01);
        let g_row = vqmovn_u16(g_row01);
        let b_row = vqmovn_u16(b_row01);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if image_configuration.has_alpha() {
            let a_row01 = vcombine_u16(vqmovn_u32(a_row0_), vqmovn_u32(a_row1_));
            let a_row = vqmovn_u16(a_row01);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x8x4_t(r_row, g_row, b_row, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x8x4_t(b_row, g_row, r_row, a_row)
                }
            };
            vst4_u8(dst_ptr, store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x8x3_t(r_row, g_row, b_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x8x3_t(b_row, g_row, r_row)
                }
            };
            vst3_u8(dst_ptr, store_rows);
        }

        cx += 8;
    }

    while cx + 4 < width as usize {
        let offset_src_ptr =
            ((src as *const u8).add(src_offset as usize) as *const f32).add(cx * channels);

        let src_ptr_0 = offset_src_ptr;

        let (r_row0_, g_row0_, b_row0_, a_row0_) =
            neon_oklab_gamma_vld::<CHANNELS_CONFIGURATION, TARGET>(
                src_ptr_0, &transfer, m0, m1, m2, m3, m4, m5, m6, m7, m8, c0, c1, c2, c3, c4, c5,
                c6, c7, c8, x0, x1, x2, x3, x4, x5, x6, x7, x8,
            );

        let zeros = vdup_n_u16(0);

        let r_row01 = vcombine_u16(vqmovn_u32(r_row0_), zeros);
        let g_row01 = vcombine_u16(vqmovn_u32(g_row0_), zeros);
        let b_row01 = vcombine_u16(vqmovn_u32(b_row0_), zeros);

        let r_row = vqmovn_u16(r_row01);
        let g_row = vqmovn_u16(g_row01);
        let b_row = vqmovn_u16(b_row01);

        let dst_ptr = dst.add(dst_offset as usize + cx * channels);

        if image_configuration.has_alpha() {
            let a_row01 = vcombine_u16(vqmovn_u32(a_row0_), zeros);
            let a_row = vqmovn_u16(a_row01);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x8x4_t(r_row, g_row, b_row, a_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x8x4_t(b_row, g_row, r_row, a_row)
                }
            };
            let mut transient: [u8; 32] = [0; 32];
            vst4_u8(transient.as_mut_ptr(), store_rows);
            std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr, 4 * 4);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    uint8x8x3_t(r_row, g_row, b_row)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    uint8x8x3_t(b_row, g_row, r_row)
                }
            };
            let mut transient: [u8; 24] = [0; 24];
            vst3_u8(transient.as_mut_ptr(), store_rows);
            std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr, 4 * 3);
        }

        cx += 4;
    }

    cx
}
