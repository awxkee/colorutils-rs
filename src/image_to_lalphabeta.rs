/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::{Rgb, TransferFunction};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
#[cfg(feature = "rayon")]
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

    #[cfg(feature = "rayon")]
    {
        let dst_slice_safe_align = unsafe {
            slice::from_raw_parts_mut(
                dst.as_mut_ptr() as *mut u8,
                dst_stride as usize * height as usize,
            )
        };
        dst_slice_safe_align
            .par_chunks_exact_mut(dst_stride as usize)
            .zip(src.par_chunks_exact(src_stride as usize))
            .for_each(|(dst, src)| unsafe {
                let mut _cx = 0usize;

                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr() as *mut f32;

                for x in _cx..width as usize {
                    let px = x * channels;

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
                    let dst_store = dst_ptr.add(px);
                    let lalphabeta = rgb.to_lalphabeta(transfer_function);
                    dst_store.write_unaligned(lalphabeta.l);
                    dst_store.add(1).write_unaligned(lalphabeta.alpha);
                    dst_store.add(2).write_unaligned(lalphabeta.beta);

                    if image_configuration.has_alpha() {
                        let a = src
                            .add(image_configuration.get_a_channel_offset())
                            .read_unaligned();
                        let a_lin = a as f32 * (1f32 / 255f32);
                        dst_store.add(3).write_unaligned(a_lin);
                    }
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;

        for _ in 0..height as usize {
            let mut _cx = 0usize;

            let src_ptr = unsafe { src.as_ptr().add(src_offset) };
            let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

            for x in _cx..width as usize {
                let px = x * channels;

                let src = unsafe { src_ptr.add(px) };
                let r = unsafe {
                    src.add(image_configuration.get_r_channel_offset())
                        .read_unaligned()
                };
                let g = unsafe {
                    src.add(image_configuration.get_g_channel_offset())
                        .read_unaligned()
                };
                let b = unsafe {
                    src.add(image_configuration.get_b_channel_offset())
                        .read_unaligned()
                };

                let rgb = Rgb::<u8>::new(r, g, b);
                let dst_store = unsafe { dst_ptr.add(px) };
                let lalphabeta = rgb.to_lalphabeta(transfer_function);
                unsafe {
                    dst_store.write_unaligned(lalphabeta.l);
                    dst_store.add(1).write_unaligned(lalphabeta.alpha);
                    dst_store.add(2).write_unaligned(lalphabeta.beta);
                }

                if image_configuration.has_alpha() {
                    let a = unsafe {
                        src.add(image_configuration.get_a_channel_offset())
                            .read_unaligned()
                    };
                    let a_lin = a as f32 * (1f32 / 255f32);
                    unsafe {
                        dst_store.add(3).write_unaligned(a_lin);
                    }
                }
            }

            src_offset += src_stride as usize;
            dst_offset += dst_stride as usize;
        }
    }
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
