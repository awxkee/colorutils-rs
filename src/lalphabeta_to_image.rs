/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::{
    ColorError, ImageBuffer, ImageBufferMut, LAlphaBeta, TransferFunction, XYZ_TO_SRGB_D65,
};

fn lalphabeta_to_image<const CHANNELS_CONFIGURATION: u8>(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    dst.validate()?;
    src.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }

    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let channels = image_configuration.channel_count();

    let mut lut_table = [0u8; 65536];
    for (i, lut) in lut_table[..2049].iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.)
            .round()
            .min(255.) as u8;
    }

    let mut transient_row = vec![0f32; src.width as usize * channels];

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;

    for (dst, src) in dst
        .data
        .borrow_mut()
        .chunks_mut(dst_stride)
        .zip(src.data.chunks(src.stride()))
    {
        let src = &src[..src_r_width as usize];
        let dst = &mut dst[..dst_r_width as usize];

        if image_configuration.has_alpha() {
            for (transient, src) in transient_row
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                let lalphabeta = LAlphaBeta::new(src[0], src[1], src[2]);
                let rgb = lalphabeta.to_linear_rgb(&XYZ_TO_SRGB_D65);
                transient[image_configuration.r_index()] = rgb.r;
                transient[image_configuration.g_index()] = rgb.g;
                transient[image_configuration.b_index()] = rgb.b;
                let a = src[3];
                transient[image_configuration.a_index()] = (a * 255f32).min(255.).round();
            }
        } else {
            for (transient, src) in transient_row
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                let lalphabeta = LAlphaBeta::new(src[0], src[1], src[2]);
                let rgb = lalphabeta.to_linear_rgb(&XYZ_TO_SRGB_D65);
                transient[image_configuration.r_index()] = rgb.r;
                transient[image_configuration.g_index()] = rgb.g;
                transient[image_configuration.b_index()] = rgb.b;
            }
        }

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(transient_row.as_chunks::<3>().0.iter())
            {
                let r = src[image_configuration.r_index()];
                let g = src[image_configuration.g_index()];
                let b = src[image_configuration.b_index()];
                let ri = (r * 2048.).round() as u16;
                let gi = (g * 2048.).round() as u16;
                let bi = (b * 2048.).round() as u16;
                dst[image_configuration.r_index()] = lut_table[ri.min(2048) as usize];
                dst[image_configuration.g_index()] = lut_table[gi.min(2048) as usize];
                dst[image_configuration.b_index()] = lut_table[bi.min(2048) as usize];
            }
        } else if image_configuration == ImageConfiguration::Rgba
            || image_configuration == ImageConfiguration::Bgra
        {
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(transient_row.as_chunks::<4>().0.iter())
            {
                let r = src[image_configuration.r_index()];
                let g = src[image_configuration.g_index()];
                let b = src[image_configuration.b_index()];
                let ri = (r * 2048.).round() as u16;
                let gi = (g * 2048.).round() as u16;
                let bi = (b * 2048.).round() as u16;
                dst[image_configuration.r_index()] = lut_table[ri.min(2048) as usize];
                dst[image_configuration.g_index()] = lut_table[gi.min(2048) as usize];
                dst[image_configuration.b_index()] = lut_table[bi.min(2048) as usize];
                dst[image_configuration.a_index()] = src[image_configuration.a_index()] as u8;
            }
        }
    }

    Ok(())
}

/// This function converts *lαβ* with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn lalphabeta_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    lalphabeta_to_image::<{ ImageConfiguration::Rgba as u8 }>(src, dst, transfer_function)
}

/// This function converts *lαβ* to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn lalphabeta_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    lalphabeta_to_image::<{ ImageConfiguration::Rgb as u8 }>(src, dst, transfer_function)
}

/// This function converts *lαβ* to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn lalphabeta_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    lalphabeta_to_image::<{ ImageConfiguration::Bgr as u8 }>(src, dst, transfer_function)
}

/// This function converts *lαβ* with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn lalphabeta_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    lalphabeta_to_image::<{ ImageConfiguration::Bgra as u8 }>(src, dst, transfer_function)
}
