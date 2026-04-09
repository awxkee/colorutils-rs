/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::{
    ColorError, ImageBuffer, ImageBufferMut, LAlphaBeta, Rgb, TransferFunction, SRGB_TO_XYZ_D65,
};

#[inline(always)]
fn channels_to_lalphabeta<const CHANNELS_CONFIGURATION: u8>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    src.validate()?;
    dst.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let mut lut_table = [0.; 256];
    for (i, element) in lut_table.iter_mut().enumerate() {
        *element = transfer_function.linearize(i as f32 * (1. / 255.0));
    }

    let mut a_lut_table = [0f32; 256];
    for (i, lut) in a_lut_table.iter_mut().enumerate() {
        *lut = i as f32 * (1. / 255.0);
    }

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

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                let r = lut_table[src[image_configuration.r_index()] as usize];
                let g = lut_table[src[image_configuration.g_index()] as usize];
                let b = lut_table[src[image_configuration.b_index()] as usize];
                let lalphabeta = LAlphaBeta::from_linear_rgb(Rgb::new(r, g, b), &SRGB_TO_XYZ_D65);
                dst[0] = lalphabeta.l;
                dst[1] = lalphabeta.alpha;
                dst[2] = lalphabeta.beta;
            }
        } else if image_configuration == ImageConfiguration::Bgra
            || image_configuration == ImageConfiguration::Rgba
        {
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                let r = lut_table[src[image_configuration.r_index()] as usize];
                let g = lut_table[src[image_configuration.g_index()] as usize];
                let b = lut_table[src[image_configuration.b_index()] as usize];
                let lalphabeta = LAlphaBeta::from_linear_rgb(Rgb::new(r, g, b), &SRGB_TO_XYZ_D65);
                dst[0] = lalphabeta.l;
                dst[1] = lalphabeta.alpha;
                dst[2] = lalphabeta.beta;
                dst[3] = a_lut_table[src[image_configuration.a_index()] as usize];
            }
        }
    }

    Ok(())
}

/// This function converts RGB to *lαβ* against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_lalphabeta(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_lalphabeta::<{ ImageConfiguration::Rgb as u8 }>(src, dst, transfer_function)
}

/// This function converts RGBA to *lαβ* against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_lalphabeta(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_lalphabeta::<{ ImageConfiguration::Rgba as u8 }>(src, dst, transfer_function)
}

/// This function converts BGRA to *lαβ* against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_lalphabeta(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_lalphabeta::<{ ImageConfiguration::Bgra as u8 }>(src, dst, transfer_function)
}

/// This function converts BGR to *lαβ* against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_lalphabeta(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_lalphabeta::<{ ImageConfiguration::Bgr as u8 }>(src, dst, transfer_function)
}
