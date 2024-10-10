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
#[cfg(not(feature = "rayon"))]
use std::slice;

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
    for i in 0..256 {
        lut_table[i] = (transfer_function.gamma(i as f32 * (1. / 255.0)) * 255.)
            .ceil()
            .min(255.) as u8;
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

                    *dst.get_unchecked_mut(px) = *lut_table.get_unchecked(rgb.r as usize);
                    *dst.get_unchecked_mut(px + 1) = *lut_table.get_unchecked(rgb.g as usize);
                    *dst.get_unchecked_mut(px + 2) = *lut_table.get_unchecked(rgb.b as usize);

                    if USE_ALPHA && image_configuration.has_alpha() {
                        let a = src.get_unchecked(px + image_configuration.get_a_channel_offset());
                        *dst.get_unchecked_mut(px + 3) = *a;
                    }
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;

        for _ in 0.._height as usize {
            let mut _cx = 0usize;

            let src_ptr = unsafe { src.as_ptr().add(src_offset) };
            let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

            let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };
            let dst_slice =
                unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

            for x in _cx..width as usize {
                let px = x * channels;
                let r = unsafe {
                    *src_slice.get_unchecked(px + image_configuration.get_r_channel_offset())
                };
                let g = unsafe {
                    *src_slice.get_unchecked(px + image_configuration.get_g_channel_offset())
                };
                let b = unsafe {
                    *src_slice.get_unchecked(px + image_configuration.get_b_channel_offset())
                };

                let rgb = Rgb::<u8>::new(r, g, b);
                let mut rgb = rgb.to_rgb_f32();

                rgb = rgb.gamma(transfer_function);

                unsafe {
                    *dst_slice.get_unchecked_mut(px) = *lut_table.get_unchecked(rgb.r as usize);
                    *dst_slice.get_unchecked_mut(px + 1) = *lut_table.get_unchecked(rgb.g as usize);
                    *dst_slice.get_unchecked_mut(px + 2) = *lut_table.get_unchecked(rgb.b as usize);
                }

                if USE_ALPHA && image_configuration.has_alpha() {
                    let a = unsafe {
                        *src_slice.get_unchecked(px + image_configuration.get_a_channel_offset())
                    };
                    unsafe {
                        *dst_slice.get_unchecked_mut(px + 3) = a;
                    }
                }
            }

            src_offset += src_stride as usize;
            dst_offset += dst_stride as usize;
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
