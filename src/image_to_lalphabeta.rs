/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::{LAlphaBeta, Rgb, TransferFunction, SRGB_TO_XYZ_D65};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

#[inline(always)]
fn channels_to_lalphabeta<const CHANNELS_CONFIGURATION: u8>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let channels = image_configuration.get_channels_count();

    let mut lut_table = vec![0f32; 256];
    for (i, element) in lut_table.iter_mut().enumerate() {
        *element = transfer_function.linearize(i as f32 * (1. / 255.0));
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

    iter.for_each(|(dst, src)| unsafe {
        let mut _cx = 0usize;

        let mut linearized_row = vec![0f32; width as usize * channels];
        for (linear_chunk, src_chunk) in linearized_row
            .chunks_exact_mut(channels)
            .zip(src.chunks_exact(channels))
        {
            linear_chunk[image_configuration.get_r_channel_offset()] = *lut_table
                .get_unchecked(src_chunk[image_configuration.get_r_channel_offset()] as usize);
            linear_chunk[image_configuration.get_g_channel_offset()] = *lut_table
                .get_unchecked(src_chunk[image_configuration.get_g_channel_offset()] as usize);
            linear_chunk[image_configuration.get_b_channel_offset()] = *lut_table
                .get_unchecked(src_chunk[image_configuration.get_b_channel_offset()] as usize);
            if image_configuration.has_alpha() {
                linear_chunk[image_configuration.get_a_channel_offset()] =
                    src_chunk[image_configuration.get_a_channel_offset()] as f32 * (1. / 255.0);
            }
        }

        let dst_ptr = dst.as_mut_ptr() as *mut f32;

        for x in _cx..width as usize {
            let px = x * channels;

            let src = linearized_row.get_unchecked(px..);
            let r = *src.get_unchecked(image_configuration.get_r_channel_offset());
            let g = *src.get_unchecked(image_configuration.get_g_channel_offset());
            let b = *src.get_unchecked(image_configuration.get_b_channel_offset());

            let rgb = Rgb::<f32>::new(r, g, b);
            let dst_store = dst_ptr.add(px);
            let lalphabeta = LAlphaBeta::from_linear_rgb(rgb, &SRGB_TO_XYZ_D65);
            dst_store.write_unaligned(lalphabeta.l);
            dst_store.add(1).write_unaligned(lalphabeta.alpha);
            dst_store.add(2).write_unaligned(lalphabeta.beta);

            if image_configuration.has_alpha() {
                let a = *src.get_unchecked(image_configuration.get_a_channel_offset());
                dst_store.add(3).write_unaligned(a);
            }
        }
    });
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_lalphabeta::<{ ImageConfiguration::Rgb as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_lalphabeta::<{ ImageConfiguration::Rgba as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_lalphabeta::<{ ImageConfiguration::Bgra as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_lalphabeta::<{ ImageConfiguration::Bgr as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
