/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_image_to_linear_unsigned::sse_channels_to_linear_u8;
use crate::Rgb;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

#[allow(clippy::type_complexity)]
fn channels_to_linear<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    l_src: &[u8],
    src_stride: u32,
    l_dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    _: u32,
    transfer_function: TransferFunction,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut _wide_row_handler: Option<
        unsafe fn(usize, *const u8, usize, u32, *mut u8, usize, TransferFunction) -> usize,
    > = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler =
            Some(neon_channels_to_linear_u8::<CHANNELS_CONFIGURATION, USE_ALPHA, true>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler =
            Some(sse_channels_to_linear_u8::<CHANNELS_CONFIGURATION, USE_ALPHA, true>);
    }

    #[cfg(not(feature = "rayon"))]
    for (dst_row, src_row) in l_dst
        .chunks_exact_mut(dst_stride as usize)
        .zip(l_src.chunks_exact(src_stride as usize))
    {
        let mut _cx = 0usize;

        if let Some(dispatcher) = _wide_row_handler {
            unsafe {
                _cx = dispatcher(
                    _cx,
                    src_row.as_ptr(),
                    0,
                    width,
                    dst_row.as_mut_ptr(),
                    0,
                    transfer_function,
                )
            }
        }

        for x in _cx..width as usize {
            let px = x * channels;
            let r =
                unsafe { *src_row.get_unchecked(px + image_configuration.get_r_channel_offset()) };
            let g =
                unsafe { *src_row.get_unchecked(px + image_configuration.get_g_channel_offset()) };
            let b =
                unsafe { *src_row.get_unchecked(px + image_configuration.get_b_channel_offset()) };

            let rgb = Rgb::<u8>::new(r, g, b);
            let mut rgb_f32 = rgb.to_rgb_f32();
            rgb_f32 = rgb_f32.linearize(transfer_function);
            let rgb = rgb_f32.to_u8();

            unsafe {
                *dst_row.get_unchecked_mut(px) = rgb.r;
                *dst_row.get_unchecked_mut(px + 1) = rgb.g;
                *dst_row.get_unchecked_mut(px + 2) = rgb.b;
            }

            if USE_ALPHA && image_configuration.has_alpha() {
                let a = unsafe {
                    *src_row.get_unchecked(px + image_configuration.get_a_channel_offset())
                };
                unsafe {
                    *dst_row.get_unchecked_mut(px + 3) = a;
                }
            }
        }
    }

    #[cfg(feature = "rayon")]
    {
        l_dst
            .par_chunks_exact_mut(dst_stride as usize)
            .zip(l_src.par_chunks_exact(src_stride as usize))
            .for_each(|(dst_row, src_row)| unsafe {
                let mut _cx = 0usize;

                if let Some(dispatcher) = _wide_row_handler {
                    _cx = dispatcher(
                        _cx,
                        src_row.as_ptr(),
                        0,
                        width,
                        dst_row.as_mut_ptr(),
                        0,
                        transfer_function,
                    )
                }

                for x in _cx..width as usize {
                    let px = x * channels;
                    let r = *src_row.get_unchecked(px + image_configuration.get_r_channel_offset());
                    let g = *src_row.get_unchecked(px + image_configuration.get_g_channel_offset());
                    let b = *src_row.get_unchecked(px + image_configuration.get_b_channel_offset());

                    let rgb = Rgb::<u8>::new(r, g, b);
                    let mut rgb_f32 = rgb.to_rgb_f32();
                    rgb_f32 = rgb_f32.linearize(transfer_function);
                    let rgb = rgb_f32.to_u8();

                    *dst_row.get_unchecked_mut(px) = rgb.r;
                    *dst_row.get_unchecked_mut(px + 1) = rgb.g;
                    *dst_row.get_unchecked_mut(px + 2) = rgb.b;

                    if USE_ALPHA && image_configuration.has_alpha() {
                        let a =
                            *src_row.get_unchecked(px + image_configuration.get_a_channel_offset());
                        *dst_row.get_unchecked_mut(px + 3) = a;
                    }
                }
            });
    }
}

/// This function converts RGB to Linear. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_linear_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
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
pub fn rgba_to_linear_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
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

/// This function converts BGRA to Linear, Alpha channel is normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgra_to_linear_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
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

/// This function converts BGR to Linear. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgr_to_linear_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
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
