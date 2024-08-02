/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

pub mod neon_image_linear_to_u8 {
    use crate::image::ImageConfiguration;
    use crate::neon::get_neon_linear_transfer;
    use crate::{load_u8_and_deinterleave, load_u8_and_deinterleave_half, TransferFunction};
    use std::arch::aarch64::*;

    #[inline(always)]
    pub(crate) unsafe fn neon_triple_to_linear_u8(
        r: uint32x4_t,
        g: uint32x4_t,
        b: uint32x4_t,
        transfer: &unsafe fn(float32x4_t) -> float32x4_t,
    ) -> (uint32x4_t, uint32x4_t, uint32x4_t) {
        let r_f = vmulq_n_f32(vcvtq_f32_u32(r), 1f32 / 255f32);
        let g_f = vmulq_n_f32(vcvtq_f32_u32(g), 1f32 / 255f32);
        let b_f = vmulq_n_f32(vcvtq_f32_u32(b), 1f32 / 255f32);
        let r_linear = vmulq_n_f32(transfer(r_f), 255f32);
        let g_linear = vmulq_n_f32(transfer(g_f), 255f32);
        let b_linear = vmulq_n_f32(transfer(b_f), 255f32);

        (
            vcvtaq_u32_f32(r_linear),
            vcvtaq_u32_f32(g_linear),
            vcvtaq_u32_f32(b_linear),
        )
    }

    #[inline]
    pub(crate) unsafe fn neon_channels_to_linear_u8<
        const CHANNELS_CONFIGURATION: u8,
        const USE_ALPHA: bool,
    >(
        start_cx: usize,
        src: *const u8,
        src_offset: usize,
        width: u32,
        dst: *mut u8,
        dst_offset: usize,
        transfer_function: TransferFunction,
    ) -> usize {
        let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
        let channels = image_configuration.get_channels_count();
        let mut cx = start_cx;

        let dst_ptr = dst.add(dst_offset);

        let transfer = get_neon_linear_transfer(transfer_function);

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

            let (x_low_low, y_low_low, z_low_low) =
                neon_triple_to_linear_u8(r_low_low, g_low_low, b_low_low, &transfer);

            let r_low_high = vmovl_high_u16(r_low);
            let g_low_high = vmovl_high_u16(g_low);
            let b_low_high = vmovl_high_u16(b_low);

            let (x_low_high, y_low_high, z_low_high) =
                neon_triple_to_linear_u8(r_low_high, g_low_high, b_low_high, &transfer);

            let r_high = vmovl_high_u8(r_chan);
            let g_high = vmovl_high_u8(g_chan);
            let b_high = vmovl_high_u8(b_chan);

            let r_high_low = vmovl_u16(vget_low_u16(r_high));
            let g_high_low = vmovl_u16(vget_low_u16(g_high));
            let b_high_low = vmovl_u16(vget_low_u16(b_high));

            let (x_high_low, y_high_low, z_high_low) =
                neon_triple_to_linear_u8(r_high_low, g_high_low, b_high_low, &transfer);

            let r_high_high = vmovl_high_u16(r_high);
            let g_high_high = vmovl_high_u16(g_high);
            let b_high_high = vmovl_high_u16(b_high);

            let (x_high_high, y_high_high, z_high_high) =
                neon_triple_to_linear_u8(r_high_high, g_high_high, b_high_high, &transfer);

            let r_u_norm = vcombine_u8(
                vqmovn_u16(vcombine_u16(vmovn_u32(x_low_low), vmovn_u32(x_low_high))),
                vqmovn_u16(vcombine_u16(vmovn_u32(x_high_low), vmovn_u32(x_high_high))),
            );

            let g_u_norm = vcombine_u8(
                vqmovn_u16(vcombine_u16(vmovn_u32(y_low_low), vmovn_u32(y_low_high))),
                vqmovn_u16(vcombine_u16(vmovn_u32(y_high_low), vmovn_u32(y_high_high))),
            );

            let b_u_norm = vcombine_u8(
                vqmovn_u16(vcombine_u16(vmovn_u32(z_low_low), vmovn_u32(z_low_high))),
                vqmovn_u16(vcombine_u16(vmovn_u32(z_high_low), vmovn_u32(z_high_high))),
            );

            if USE_ALPHA {
                let v_4 = uint8x16x4_t(r_u_norm, g_u_norm, b_u_norm, a_chan);
                vst4q_u8(dst_ptr.add(cx * channels), v_4);
            } else {
                let v_4 = uint8x16x3_t(r_u_norm, g_u_norm, b_u_norm);
                vst3q_u8(dst_ptr.add(cx * channels), v_4);
            }

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

            let (x_low_low, y_low_low, z_low_low) =
                neon_triple_to_linear_u8(r_low_low, g_low_low, b_low_low, &transfer);

            let r_low_high = vmovl_high_u16(r_low);
            let g_low_high = vmovl_high_u16(g_low);
            let b_low_high = vmovl_high_u16(b_low);

            let (x_low_high, y_low_high, z_low_high) =
                neon_triple_to_linear_u8(r_low_high, g_low_high, b_low_high, &transfer);

            let r_u_norm = vqmovn_u16(vcombine_u16(vmovn_u32(x_low_low), vmovn_u32(x_low_high)));

            let g_u_norm = vqmovn_u16(vcombine_u16(vmovn_u32(y_low_low), vmovn_u32(y_low_high)));

            let b_u_norm = vqmovn_u16(vcombine_u16(vmovn_u32(z_low_low), vmovn_u32(z_low_high)));

            if USE_ALPHA {
                let v_4 = uint8x8x4_t(r_u_norm, g_u_norm, b_u_norm, vget_low_u8(a_chan));
                vst4_u8(dst_ptr.add(cx * channels), v_4);
            } else {
                let v_4 = uint8x8x3_t(r_u_norm, g_u_norm, b_u_norm);
                vst3_u8(dst_ptr.add(cx * channels), v_4);
            }

            cx += 8;
        }

        cx
    }
}
