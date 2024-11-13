/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::Rgb;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

#[allow(clippy::type_complexity)]
fn linear_to_gamma_channels<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    _height: u32,
    transfer_function: TransferFunction,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut lut_table = vec![0u8; 256];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 255.0)) * 255.).min(255.) as u8;
    }

    #[cfg(feature = "rayon")]
    {
        dst.par_chunks_exact_mut(dst_stride as usize)
            .zip(src.par_chunks_exact(src_stride as usize))
            .for_each(|(dst, src)| unsafe {
                let mut _cx = 0usize;

                for x in _cx..width as usize {
                    let px = x * channels;
                    let r = *src.get_unchecked(px + image_configuration.get_r_channel_offset());
                    let g = *src.get_unchecked(px + image_configuration.get_g_channel_offset());
                    let b = *src.get_unchecked(px + image_configuration.get_b_channel_offset());

                    let rgb = Rgb::<u8>::new(r, g, b);

                    let dst = dst.get_unchecked_mut(px..);

                    *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) =
                        *lut_table.get_unchecked(rgb.r as usize);
                    *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) =
                        *lut_table.get_unchecked(rgb.g as usize);
                    *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) =
                        *lut_table.get_unchecked(rgb.b as usize);

                    if USE_ALPHA && image_configuration.has_alpha() {
                        let a = src.get_unchecked(px + image_configuration.get_a_channel_offset());
                        *dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) = *a;
                    }
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for (dst, src) in dst
            .chunks_exact_mut(dst_stride as usize)
            .zip(src.chunks_exact(src_stride as usize))
        {
            unsafe {
                let mut _cx = 0usize;

                for x in _cx..width as usize {
                    let px = x * channels;
                    let r = *src.get_unchecked(px + image_configuration.get_r_channel_offset());
                    let g = *src.get_unchecked(px + image_configuration.get_g_channel_offset());
                    let b = *src.get_unchecked(px + image_configuration.get_b_channel_offset());

                    let rgb = Rgb::<u8>::new(r, g, b);

                    let dst = dst.get_unchecked_mut(px..);

                    *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) =
                        *lut_table.get_unchecked(rgb.r as usize);
                    *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) =
                        *lut_table.get_unchecked(rgb.g as usize);
                    *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) =
                        *lut_table.get_unchecked(rgb.b as usize);

                    if USE_ALPHA && image_configuration.has_alpha() {
                        let a = src.get_unchecked(px + image_configuration.get_a_channel_offset());
                        *dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) = *a;
                    }
                }
            }
        }
    }
}

/// This function converts Linear to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive Gamma RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_u8_to_rgb(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels::<{ ImageConfiguration::Rgb as u8 }, false>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Linear RGBA to RGBA, Alpha channel will be denormalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_u8_to_rgba(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels::<{ ImageConfiguration::Rgba as u8 }, true>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Linear BGRA to BGRA, Alpha channel will de dernormalizaed. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive Gamma BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_u8_to_bgra(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels::<{ ImageConfiguration::Bgra as u8 }, true>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Linear BGR to Gamma BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_u8_to_bgr(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels::<{ ImageConfiguration::Bgr as u8 }, false>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
