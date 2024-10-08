/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::neon::*;
use crate::{
    load_u8_and_deinterleave, load_u8_and_deinterleave_half, load_u8_and_deinterleave_quarter,
};
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_triple_to_linear(
    r: uint32x4_t,
    g: uint32x4_t,
    b: uint32x4_t,
    transfer_function: TransferFunction,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let r_f = vmulq_n_f32(vcvtq_f32_u32(r), 1f32 / 255f32);
    let g_f = vmulq_n_f32(vcvtq_f32_u32(g), 1f32 / 255f32);
    let b_f = vmulq_n_f32(vcvtq_f32_u32(b), 1f32 / 255f32);
    let r_linear = neon_perform_linear_transfer(transfer_function, r_f);
    let g_linear = neon_perform_linear_transfer(transfer_function, g_f);
    let b_linear = neon_perform_linear_transfer(transfer_function, b_f);
    (r_linear, g_linear, b_linear)
}

pub unsafe fn neon_channels_to_linear<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut f32,
    dst_offset: usize,
    transfer_function: TransferFunction,
) -> usize {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.get_channels_count();
    let mut cx = start_cx;

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

        let (x_low_low, y_low_low, z_low_low) =
            neon_triple_to_linear(r_low_low, g_low_low, b_low_low, transfer_function);

        let a_low = vmovl_u8(vget_low_u8(a_chan));

        if USE_ALPHA {
            let a_low_low =
                vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_low))), 1f32 / 255f32);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(x_low_low, y_low_low, z_low_low, a_low_low)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(z_low_low, y_low_low, x_low_low, a_low_low)
                }
            };
            vst4q_f32(dst_ptr.add(cx * channels), store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(x_low_low, y_low_low, z_low_low)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(z_low_low, y_low_low, x_low_low)
                }
            };
            vst3q_f32(dst_ptr.add(cx * channels), store_rows);
        }

        let r_low_high = vmovl_high_u16(r_low);
        let g_low_high = vmovl_high_u16(g_low);
        let b_low_high = vmovl_high_u16(b_low);

        let (x_low_high, y_low_high, z_low_high) =
            neon_triple_to_linear(r_low_high, g_low_high, b_low_high, transfer_function);

        if USE_ALPHA {
            let a_low_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_low)), 1f32 / 255f32);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(x_low_high, y_low_high, z_low_high, a_low_high)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(z_low_high, y_low_high, x_low_high, a_low_high)
                }
            };
            vst4q_f32(dst_ptr.add(cx * channels + 4 * channels), store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(x_low_high, y_low_high, z_low_high)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(z_low_high, y_low_high, x_low_high)
                }
            };
            vst3q_f32(dst_ptr.add(cx * channels + 4 * channels), store_rows);
        }

        let r_high = vmovl_high_u8(r_chan);
        let g_high = vmovl_high_u8(g_chan);
        let b_high = vmovl_high_u8(b_chan);

        let r_high_low = vmovl_u16(vget_low_u16(r_high));
        let g_high_low = vmovl_u16(vget_low_u16(g_high));
        let b_high_low = vmovl_u16(vget_low_u16(b_high));

        let (x_high_low, y_high_low, z_high_low) =
            neon_triple_to_linear(r_high_low, g_high_low, b_high_low, transfer_function);

        let a_high = vmovl_high_u8(a_chan);

        if USE_ALPHA {
            let a_high_low = vmulq_n_f32(
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_high))),
                1f32 / 255f32,
            );
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(x_high_low, y_high_low, z_high_low, a_high_low)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(z_high_low, y_high_low, x_high_low, a_high_low)
                }
            };
            vst4q_f32(dst_ptr.add(cx * channels + 4 * channels * 2), store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(x_high_low, y_high_low, z_high_low)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(z_high_low, y_high_low, x_high_low)
                }
            };
            vst3q_f32(dst_ptr.add(cx * channels + 4 * channels * 2), store_rows);
        }

        let r_high_high = vmovl_high_u16(r_high);
        let g_high_high = vmovl_high_u16(g_high);
        let b_high_high = vmovl_high_u16(b_high);

        let (x_high_high, y_high_high, z_high_high) =
            neon_triple_to_linear(r_high_high, g_high_high, b_high_high, transfer_function);

        if USE_ALPHA {
            let a_high_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_high)), 1f32 / 255f32);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(x_high_high, y_high_high, z_high_high, a_high_high)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(z_high_high, y_high_high, x_high_high, a_high_high)
                }
            };
            vst4q_f32(dst_ptr.add(cx * channels + 4 * channels * 3), store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(x_high_high, y_high_high, z_high_high)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(z_high_high, y_high_high, x_high_high)
                }
            };
            vst3q_f32(dst_ptr.add(cx * channels + 4 * channels * 3), store_rows);
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
            neon_triple_to_linear(r_low_low, g_low_low, b_low_low, transfer_function);

        let a_low = vmovl_u8(vget_low_u8(a_chan));

        if USE_ALPHA {
            let a_low_low =
                vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_low))), 1f32 / 255f32);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(x_low_low, y_low_low, z_low_low, a_low_low)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(z_low_low, y_low_low, x_low_low, a_low_low)
                }
            };
            vst4q_f32(dst_ptr.add(cx * channels), store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(x_low_low, y_low_low, z_low_low)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(z_low_low, y_low_low, x_low_low)
                }
            };
            vst3q_f32(dst_ptr.add(cx * channels), store_rows);
        }

        let r_low_high = vmovl_high_u16(r_low);
        let g_low_high = vmovl_high_u16(g_low);
        let b_low_high = vmovl_high_u16(b_low);

        let (x_low_high, y_low_high, z_low_high) =
            neon_triple_to_linear(r_low_high, g_low_high, b_low_high, transfer_function);

        if USE_ALPHA {
            let a_low_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(a_low)), 1f32 / 255f32);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(x_low_high, y_low_high, z_low_high, a_low_high)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(z_low_high, y_low_high, x_low_high, a_low_high)
                }
            };
            vst4q_f32(dst_ptr.add(cx * channels + 4 * channels), store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(x_low_high, y_low_high, z_low_high)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(z_low_high, y_low_high, x_low_high)
                }
            };
            vst3q_f32(dst_ptr.add(cx * channels + 4 * channels), store_rows);
        }

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

        let (x_low_low, y_low_low, z_low_low) =
            neon_triple_to_linear(r_low_low, g_low_low, b_low_low, transfer_function);

        let a_low = vmovl_u8(vget_low_u8(a_chan));

        if USE_ALPHA {
            let a_low_low =
                vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_low))), 1f32 / 255f32);
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x4_t(x_low_low, y_low_low, z_low_low, a_low_low)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x4_t(z_low_low, y_low_low, x_low_low, a_low_low)
                }
            };
            vst4q_f32(dst_ptr.add(cx * channels), store_rows);
        } else {
            let store_rows = match image_configuration {
                ImageConfiguration::Rgb | ImageConfiguration::Rgba => {
                    float32x4x3_t(x_low_low, y_low_low, z_low_low)
                }
                ImageConfiguration::Bgra | ImageConfiguration::Bgr => {
                    float32x4x3_t(z_low_low, y_low_low, x_low_low)
                }
            };
            vst3q_f32(dst_ptr.add(cx * channels), store_rows);
        }

        cx += 4;
    }

    cx
}
