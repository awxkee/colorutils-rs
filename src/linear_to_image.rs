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
use std::slice;

#[allow(clippy::type_complexity)]
fn linear_to_gamma_channels<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut lut_table = vec![0u8; 2049];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.).min(255.) as u8;
    }

    let src_slice_safe_align = unsafe {
        slice::from_raw_parts(
            src.as_ptr() as *const u8,
            src_stride as usize * height as usize,
        )
    };

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = dst
            .par_chunks_exact_mut(dst_stride as usize)
            .zip(src_slice_safe_align.par_chunks_exact(src_stride as usize));
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = dst
            .chunks_exact_mut(dst_stride as usize)
            .zip(src_slice_safe_align.chunks_exact(src_stride as usize));
    }

    iter.for_each(|(dst, src)| unsafe {
        let mut _cx = 0usize;

        let src_ptr = src.as_ptr() as *const f32;
        let dst_ptr = dst.as_mut_ptr();

        for x in _cx..width as usize {
            let px = x * channels;
            let src_slice = src_ptr.add(px);
            let r = src_slice
                .add(image_configuration.get_r_channel_offset())
                .read_unaligned();
            let g = src_slice
                .add(image_configuration.get_g_channel_offset())
                .read_unaligned();
            let b = src_slice
                .add(image_configuration.get_b_channel_offset())
                .read_unaligned();

            let rgb = (Rgb::<f32>::new(
                r.min(1f32).max(0f32),
                g.min(1f32).max(0f32),
                b.min(1f32).max(0f32),
            ) * Rgb::<f32>::dup(2048f32))
            .round()
            .cast::<u16>();

            let dst = dst_ptr.add(px);

            dst.add(image_configuration.get_r_channel_offset())
                .write_unaligned(*lut_table.get_unchecked(rgb.r.min(2048) as usize));
            dst.add(image_configuration.get_g_channel_offset())
                .write_unaligned(*lut_table.get_unchecked(rgb.g.min(2048) as usize));
            dst.add(image_configuration.get_b_channel_offset())
                .write_unaligned(*lut_table.get_unchecked(rgb.b.min(2048) as usize));

            if USE_ALPHA && image_configuration.has_alpha() {
                let a = src_slice
                    .add(image_configuration.get_a_channel_offset())
                    .read_unaligned();
                let a_lin = (a * 255f32).round() as u8;
                dst.add(image_configuration.get_a_channel_offset())
                    .write_unaligned(a_lin);
            }
        }
    });
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
pub fn linear_to_rgb(
    src: &[f32],
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
pub fn linear_to_rgba(
    src: &[f32],
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
pub fn linear_to_bgra(
    src: &[f32],
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
pub fn linear_to_bgr(
    src: &[f32],
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
