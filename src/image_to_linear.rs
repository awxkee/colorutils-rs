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
fn channels_to_linear<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
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

    let mut lut_table = vec![0f32; 256];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = transfer_function.linearize(i as f32 * (1. / 255.0));
    }

    let dst_slice_safe_align = unsafe {
        slice::from_raw_parts_mut(
            dst.as_mut_ptr() as *mut u8,
            dst_stride as usize * height as usize,
        )
    };

    let iter;

    #[cfg(feature = "rayon")]
    {
        iter = dst_slice_safe_align
            .par_chunks_exact_mut(dst_stride as usize)
            .zip(src.par_chunks_exact(src_stride as usize));
    }

    #[cfg(not(feature = "rayon"))]
    {
        iter = dst_slice_safe_align
            .chunks_exact_mut(dst_stride as usize)
            .zip(src.chunks_exact(src_stride as usize));
    }

    iter.for_each(|(dst_row, src_row)| unsafe {
        let mut _cx = 0usize;

        let src_ptr = src_row.as_ptr();
        let dst_ptr = dst_row.as_mut_ptr() as *mut f32;

        for x in _cx..width as usize {
            let px = x * channels;
            let dst = dst_ptr.add(px);
            let src = src_ptr.add(px);
            let r = src
                .add(image_configuration.get_r_channel_offset())
                .read_unaligned();
            let g = src
                .add(image_configuration.get_g_channel_offset())
                .read_unaligned();
            let b = src
                .add(image_configuration.get_b_channel_offset())
                .read_unaligned();

            let rgb = Rgb::<u8>::new(r, g, b);

            dst.add(image_configuration.get_r_channel_offset())
                .write_unaligned(*lut_table.get_unchecked(rgb.r as usize));
            dst.add(image_configuration.get_g_channel_offset())
                .write_unaligned(*lut_table.get_unchecked(rgb.g as usize));
            dst.add(image_configuration.get_b_channel_offset())
                .write_unaligned(*lut_table.get_unchecked(rgb.b as usize));

            if USE_ALPHA && image_configuration.has_alpha() {
                let a = src
                    .add(image_configuration.get_a_channel_offset())
                    .read_unaligned();
                let a_lin = a as f32 * (1f32 / 255f32);
                dst.add(image_configuration.get_a_channel_offset())
                    .write_unaligned(a_lin);
            }
        }
    });
}

/// This function converts RGB to linear colorspace
///
/// This function converts RGB to linear color space. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_linear(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear::<{ ImageConfiguration::Rgb as u8 }, false>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts RGBA to liner color space
///
/// This function converts RGBA to Linear, Alpha channel is normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgba_to_linear(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear::<{ ImageConfiguration::Rgba as u8 }, true>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts BGRA to Linear.
///
/// This function converts BGRA to Linear, alpha channel is normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgra_to_linear(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear::<{ ImageConfiguration::Bgra as u8 }, true>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts BGR to linear
///
/// This function converts BGR to linear color space. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgr_to_linear(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear::<{ ImageConfiguration::Bgr as u8 }, false>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
