/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::neon::cie::{
    neon_triple_to_lab, neon_triple_to_lch, neon_triple_to_luv, neon_triple_to_xyz,
};
use crate::neon::*;
use crate::xyz_target::XyzTarget;
use crate::{
    load_u8_and_deinterleave, load_u8_and_deinterleave_half, load_u8_and_deinterleave_quarter,
};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_channels_to_xyza_or_laba<
    const CHANNELS_CONFIGURATION: u8,
    const TARGET: u8,
    const TRANSFER_FUNCTION: u8,
>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    matrix: &[[f32; 3]; 3],
    _: TransferFunction,
) -> usize {
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

    let transfer_function: TransferFunction = TRANSFER_FUNCTION.into();
    let transfer = get_neon_linear_transfer(transfer_function);

    let cq1 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(0));
    let cq2 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(1));
    let cq3 = vdupq_n_f32(*matrix.get_unchecked(0).get_unchecked(2));
    let cq4 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(0));
    let cq5 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(1));
    let cq6 = vdupq_n_f32(*matrix.get_unchecked(1).get_unchecked(2));
    let cq7 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(0));
    let cq8 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(1));
    let cq9 = vdupq_n_f32(*matrix.get_unchecked(2).get_unchecked(2));

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut f32;

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_u8_and_deinterleave!(src_ptr, image_configuration);

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
            XyzTarget::LCH => {
                let (l, c, h) = neon_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
            XyzTarget::LUV => {
                let (l, u, v) = neon_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
        }

        let a_low = vmovl_u8(vget_low_u8(a_chan));

        let a_low_low = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_low))), 1f32 / 255f32);

        let xyz_low_low = float32x4x4_t(x_low_low, y_low_low, z_low_low, a_low_low);
        vst4q_f32(dst_ptr.add(cx * 4), xyz_low_low);

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
            XyzTarget::LCH => {
                let (l, c, h) = neon_triple_to_lch(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = c;
                z_low_high = h;
            }
        }

        let a_low_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_low)), 1f32 / 255f32);

        let xyz_low_low = float32x4x4_t(x_low_high, y_low_high, z_low_high, a_low_high);
        vst4q_f32(dst_ptr.add(cx * 4 + 4 * 4), xyz_low_low);

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
            XyzTarget::LCH => {
                let (l, c, h) = neon_triple_to_lch(x_high_low, y_high_low, z_high_low);
                x_high_low = l;
                y_high_low = c;
                z_high_low = h;
            }
        }

        let a_high = vmovl_high_u8(a_chan);
        let a_high_low = vmulq_n_f32(
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_high))),
            1f32 / 255f32,
        );

        let xyz_low_low = float32x4x4_t(x_high_low, y_high_low, z_high_low, a_high_low);
        vst4q_f32(dst_ptr.add(cx * 4 + 4 * 4 * 2), xyz_low_low);

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
            XyzTarget::LCH => {
                let (l, c, h) = neon_triple_to_lch(x_high_high, y_high_high, z_high_high);
                x_high_high = l;
                y_high_high = c;
                z_high_high = h;
            }
        }

        let a_high_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_high)), 1f32 / 255f32);

        let xyz_low_low = float32x4x4_t(x_high_high, y_high_high, z_high_high, a_high_high);
        vst4q_f32(dst_ptr.add(cx * 4 + 4 * 4 * 3), xyz_low_low);

        cx += 16;
    }

    while cx + 8 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_u8_and_deinterleave_half!(src_ptr, image_configuration);

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
            XyzTarget::LCH => {
                let (l, c, h) = neon_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
            XyzTarget::LUV => {
                let (l, u, v) = neon_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
        }

        let a_low = vmovl_u8(vget_low_u8(a_chan));

        let a_low_low = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_low))), 1f32 / 255f32);

        let xyz_low_low = float32x4x4_t(x_low_low, y_low_low, z_low_low, a_low_low);
        vst4q_f32(dst_ptr.add(cx * 4), xyz_low_low);

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
            XyzTarget::LCH => {
                let (l, c, h) = neon_triple_to_lch(x_low_high, y_low_high, z_low_high);
                x_low_high = l;
                y_low_high = c;
                z_low_high = h;
            }
        }

        let a_low_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_low)), 1f32 / 255f32);

        let xyz_low_low = float32x4x4_t(x_low_high, y_low_high, z_low_high, a_low_high);
        vst4q_f32(dst_ptr.add(cx * 4 + 4 * 4), xyz_low_low);

        cx += 8;
    }

    while cx + 4 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_u8_and_deinterleave_quarter!(src_ptr, image_configuration);

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
            XyzTarget::LCH => {
                let (l, c, h) = neon_triple_to_lch(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = c;
                z_low_low = h;
            }
            XyzTarget::LUV => {
                let (l, u, v) = neon_triple_to_luv(x_low_low, y_low_low, z_low_low);
                x_low_low = l;
                y_low_low = u;
                z_low_low = v;
            }
        }

        let a_low = vmovl_u8(vget_low_u8(a_chan));

        let a_low_low = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_low))), 1f32 / 255f32);

        let xyz_low_low = float32x4x4_t(x_low_low, y_low_low, z_low_low, a_low_low);
        vst4q_f32(dst_ptr.add(cx * 4), xyz_low_low);

        cx += 4;
    }

    cx
}
