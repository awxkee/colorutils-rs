/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_channels_to_linear;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_channels_to_linear;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::*;
use crate::Rgb;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
#[cfg(feature = "rayon")]
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

    let mut _wide_row_handle: Option<
        unsafe fn(usize, *const u8, usize, u32, *mut f32, usize, TransferFunction) -> usize,
    > = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_channels_to_linear::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handle = Some(avx_channels_to_linear::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handle = Some(neon_channels_to_linear::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;
        for _ in 0..height as usize {
            let mut _cx = 0usize;

            if let Some(dispatcher) = _wide_row_handle {
                unsafe {
                    _cx = dispatcher(
                        _cx,
                        src.as_ptr(),
                        src_offset,
                        width,
                        dst.as_mut_ptr(),
                        dst_offset,
                        transfer_function,
                    )
                }
            }

            let src_ptr = unsafe { src.as_ptr().add(src_offset) };
            let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

            for x in _cx..width as usize {
                let px = x * channels;
                let dst = unsafe { dst_ptr.add(px) };
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
                let rgb_f32 = rgb.to_rgb_f32();

                unsafe {
                    dst.write_unaligned(transfer_function.linearize(rgb_f32.r));
                    dst.add(1)
                        .write_unaligned(transfer_function.linearize(rgb_f32.g));
                    dst.add(2)
                        .write_unaligned(transfer_function.linearize(rgb_f32.b));
                }

                if USE_ALPHA && image_configuration.has_alpha() {
                    let a = unsafe {
                        src.add(image_configuration.get_a_channel_offset())
                            .read_unaligned()
                    };
                    let a_lin = a as f32 * (1f32 / 255f32);
                    unsafe {
                        dst.add(3).write_unaligned(a_lin);
                    }
                }
            }

            src_offset += src_stride as usize;
            dst_offset += dst_stride as usize;
        }
    }

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
            .for_each(|(dst_row, src_row)| unsafe {
                let mut _cx = 0usize;

                if let Some(dispatcher) = _wide_row_handle {
                    _cx = dispatcher(
                        _cx,
                        src_row.as_ptr(),
                        0,
                        width,
                        dst_row.as_mut_ptr() as *mut f32,
                        0,
                        transfer_function,
                    )
                }

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
                    let rgb_f32 = rgb.to_rgb_f32();

                    dst.write_unaligned(transfer_function.linearize(rgb_f32.r));
                    dst.add(1)
                        .write_unaligned(transfer_function.linearize(rgb_f32.g));
                    dst.add(2)
                        .write_unaligned(transfer_function.linearize(rgb_f32.b));

                    if USE_ALPHA && image_configuration.has_alpha() {
                        let a = src
                            .add(image_configuration.get_a_channel_offset())
                            .read_unaligned();
                        let a_lin = a as f32 * (1f32 / 255f32);
                        dst.add(3).write_unaligned(a_lin);
                    }
                }
            });
    }
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
