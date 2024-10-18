/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::{LAlphaBeta, Rgb, TransferFunction, XYZ_TO_SRGB_D65};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

fn lalphabeta_to_image<const CHANNELS_CONFIGURATION: u8>(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let channels = image_configuration.get_channels_count();

    let mut lut_table = vec![0u8; 2049];
    for i in 0..2049 {
        lut_table[i] = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.).min(255.) as u8;
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

        let src_ptr = src.as_ptr() as *mut f32;

        let mut transient_row = vec![0f32; width as usize * channels];

        for x in _cx..width as usize {
            let px = x * channels;
            let l_x = src_ptr.add(px).read_unaligned();
            let l_y = src_ptr.add(px + 1).read_unaligned();
            let l_z = src_ptr.add(px + 2).read_unaligned();
            let lalphabeta = LAlphaBeta::new(l_x, l_y, l_z);
            let rgb = lalphabeta.to_linear_rgb(&XYZ_TO_SRGB_D65);

            let dst = transient_row.get_unchecked_mut((x * channels)..);
            *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) = rgb.r;
            *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) = rgb.g;
            *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) = rgb.b;
            if image_configuration.has_alpha() {
                let l_a = src_ptr.add(px + 3).read_unaligned();
                let a_value = (l_a * 255f32).max(0f32).round();
                *dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) = a_value;
            }
        }

        for (dst, src) in dst
            .chunks_exact_mut(channels)
            .zip(transient_row.chunks_exact(channels))
        {
            let r = src[image_configuration.get_r_channel_offset()];
            let g = src[image_configuration.get_g_channel_offset()];
            let b = src[image_configuration.get_b_channel_offset()];

            let rgb = (Rgb::<f32>::new(
                r.min(1f32).max(0f32),
                g.min(1f32).max(0f32),
                b.min(1f32).max(0f32),
            ) * Rgb::<f32>::dup(2048f32))
            .round()
            .cast::<u16>();

            *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) =
                *lut_table.get_unchecked(rgb.r.min(2048) as usize);
            *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) =
                *lut_table.get_unchecked(rgb.g.min(2048) as usize);
            *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) =
                *lut_table.get_unchecked(rgb.b.min(2048) as usize);
            if image_configuration.has_alpha() {
                *dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) =
                    *src.get_unchecked(image_configuration.get_a_channel_offset()) as u8;
            }
        }
    });
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
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    lalphabeta_to_image::<{ ImageConfiguration::Rgba as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
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
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    lalphabeta_to_image::<{ ImageConfiguration::Rgb as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
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
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    lalphabeta_to_image::<{ ImageConfiguration::Bgr as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
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
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    lalphabeta_to_image::<{ ImageConfiguration::Bgra as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
