/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_from_sigmoidal_row;
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_from_sigmoidal_row;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_from_sigmoidal_row;
use crate::{Rgb, Sigmoidal};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
#[cfg(feature = "rayon")]
use std::slice;

#[allow(clippy::type_complexity)]
fn sigmoidal_to_image<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut _wide_row_handler: Option<unsafe fn(usize, *const f32, *mut u8, u32) -> usize> = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler = Some(sse_from_sigmoidal_row::<CHANNELS_CONFIGURATION>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handler = Some(avx_from_sigmoidal_row::<CHANNELS_CONFIGURATION>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler = Some(neon_from_sigmoidal_row::<CHANNELS_CONFIGURATION>);
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

        if let Some(dispatcher) = _wide_row_handler {
            _cx = dispatcher(_cx, src_ptr, dst_ptr, width);
        }

        for x in _cx..width as usize {
            let px = x * channels;
            let reading_ptr = src_ptr.add(px);
            let sr = reading_ptr.read_unaligned();
            let sg = reading_ptr.add(1).read_unaligned();
            let sb = reading_ptr.add(2).read_unaligned();

            let sigmoidal = Sigmoidal::new(sr, sg, sb);
            let rgb: Rgb<u8> = sigmoidal.into();

            let hx = x * channels;

            let dst = dst_ptr.add(hx);

            dst.add(image_configuration.get_r_channel_offset())
                .write_unaligned(rgb.r);
            dst.add(image_configuration.get_g_channel_offset())
                .write_unaligned(rgb.g);
            dst.add(image_configuration.get_b_channel_offset())
                .write_unaligned(rgb.b);

            if image_configuration.has_alpha() {
                let a = (reading_ptr.add(3).read_unaligned() * 255f32)
                    .max(0f32)
                    .round()
                    .min(255f32);
                dst.add(image_configuration.get_a_channel_offset())
                    .write_unaligned(a as u8);
            }
        }
    });
}

/// This function converts Sigmoid to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    sigmoidal_to_image::<{ ImageConfiguration::Rgb as u8 }, false>(
        src, src_stride, dst, dst_stride, width, height,
    );
}

/// This function converts Sigmoid to BGRA. Alpha channel expected to be normalized and will be denormalized during transformation. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    sigmoidal_to_image::<{ ImageConfiguration::Bgra as u8 }, true>(
        src, src_stride, dst, dst_stride, width, height,
    );
}

/// This function converts Sigmoid to RGBA. Alpha channel expected to be normalized and will be denormalized during transformation. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    sigmoidal_to_image::<{ ImageConfiguration::Rgba as u8 }, true>(
        src, src_stride, dst, dst_stride, width, height,
    );
}
