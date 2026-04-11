/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::xyz_lab_to_image::xyz_target_to_rgb;
use crate::xyz_target::XyzTarget;
use crate::{ColorError, ImageBuffer, ImageBufferMut};

fn xyz_with_alpha_to_channels<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    dst.validate()?;
    src.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }

    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let source: XyzTarget = TARGET.into();

    if !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut _wide_row_handler: Option<
        unsafe fn(usize, &[f32], &mut [f32], u32, &[[f32; 3]; 3]) -> usize,
    > = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::neon::neon_xyza_to_image;
        _wide_row_handler = Some(neon_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        use crate::sse::sse_xyza_to_image;
        _wide_row_handler = Some(sse_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        use crate::avx::avx_xyza_to_image;
        _wide_row_handler = Some(avx_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let channels = image_configuration.channel_count();

    let mut lut_table = [0u8; 65536];
    for (i, lut) in lut_table[..2049].iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.)
            .round()
            .min(255.) as u8;
    }

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;
    let width = src.width;

    let mut transient_row = vec![0f32; width as usize * channels];

    for (dst, src) in dst
        .data
        .borrow_mut()
        .chunks_mut(dst_stride)
        .zip(src.data.chunks(src.stride()))
    {
        let src = &src[..src_r_width as usize];
        let dst = &mut dst[..dst_r_width as usize];

        let mut cx = 0usize;
        if let Some(dispatcher) = _wide_row_handler {
            cx = unsafe { dispatcher(0, src, &mut transient_row, width, matrix) };
        }

        for (transient, src) in transient_row[cx * channels..]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(src[cx * 4..].as_chunks::<4>().0.iter())
        {
            let rgb = xyz_target_to_rgb(source, src, matrix);
            transient[image_configuration.r_index()] = rgb.r;
            transient[image_configuration.g_index()] = rgb.g;
            transient[image_configuration.b_index()] = rgb.b;
            transient[image_configuration.a_index()] = src[3];
        }

        for (dst, src) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(transient_row.as_chunks::<4>().0.iter())
        {
            let r = (src[image_configuration.r_index()] * 2048.).round() as u16;
            let g = (src[image_configuration.g_index()] * 2048.).round() as u16;
            let b = (src[image_configuration.b_index()] * 2048.).round() as u16;
            dst[image_configuration.r_index()] = lut_table[r.min(2048) as usize];
            dst[image_configuration.g_index()] = lut_table[g.min(2048) as usize];
            dst[image_configuration.b_index()] = lut_table[b.min(2048) as usize];
            dst[image_configuration.a_index()] = (src[image_configuration.a_index()] * 255.)
                .clamp(0., 255.)
                .round() as u8;
        }
    }

    Ok(())
}

pub fn lab_with_alpha_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

pub fn lab_with_alpha_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

pub fn luv_with_alpha_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

pub fn luv_with_alpha_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

pub fn xyz_with_alpha_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

pub fn xyz_with_alpha_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

pub fn lch_with_alpha_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

pub fn lch_with_alpha_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}
