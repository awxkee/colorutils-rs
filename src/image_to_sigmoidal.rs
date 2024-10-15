/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_image_to_sigmoidal_row;

use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_image_to_sigmoidal;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_image_to_sigmoidal_row;
use crate::Rgb;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

#[allow(clippy::type_complexity)]
fn image_to_sigmoidal<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut _wide_row_handler: Option<unsafe fn(usize, *const u8, u32, *mut f32) -> usize> = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler = Some(neon_image_to_sigmoidal::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler = Some(sse_image_to_sigmoidal_row::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handler = Some(avx_image_to_sigmoidal_row::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    const COLOR_SCALE: f32 = 1f32 / 255f32;

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

        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr() as *mut f32;

        if let Some(dispatcher) = _wide_row_handler {
            _cx = dispatcher(_cx, src_ptr, width, dst_ptr);
        }

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

            let writing_ptr = dst_ptr.add(px);

            let sigmoidal = rgb.to_sigmoidal();
            writing_ptr.write_unaligned(sigmoidal.sr);
            writing_ptr.add(1).write_unaligned(sigmoidal.sg);
            writing_ptr.add(2).write_unaligned(sigmoidal.sb);

            if image_configuration.has_alpha() {
                let a = src
                    .add(image_configuration.get_a_channel_offset())
                    .read_unaligned() as f32
                    * COLOR_SCALE;

                writing_ptr.add(3).write_unaligned(a);
            }
        }
    });
}

/// This function converts RGB to Sigmoidal. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
pub fn rgb_to_sigmoidal(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_sigmoidal::<{ ImageConfiguration::Rgb as u8 }, false>(
        src, src_stride, dst, dst_stride, width, height,
    );
}

/// This function converts BGRA to Sigmoidal. Alpha channel will be normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Sigmodal data
/// * `dst_stride` - Bytes per row for dst data
pub fn bgra_to_sigmoidal(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_sigmoidal::<{ ImageConfiguration::Bgra as u8 }, true>(
        src, src_stride, dst, dst_stride, width, height,
    );
}

/// This function converts RGBA to Sigmoidal. Alpha channel will be normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Sigmoidal data
/// * `dst_stride` - Bytes per row for dst data
pub fn rgba_to_sigmoidal(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_sigmoidal::<{ ImageConfiguration::Rgba as u8 }, true>(
        src, src_stride, dst, dst_stride, width, height,
    );
}
