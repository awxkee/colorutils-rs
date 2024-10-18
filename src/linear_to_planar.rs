/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::TransferFunction;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

#[allow(clippy::type_complexity)]
fn linear_to_gamma_channels(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
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

    #[cfg(feature = "rayon")]
    {
        dst.par_chunks_exact_mut(dst_stride as usize)
            .zip(src_slice_safe_align.par_chunks_exact(src_stride as usize))
            .for_each(|(dst, src)| unsafe {
                let mut _cx = 0usize;

                let src_ptr = src.as_ptr() as *const f32;
                let dst_ptr = dst.as_mut_ptr();

                for x in _cx..width as usize {
                    let px = x;
                    let src_slice = src_ptr.add(px);
                    let pixel =
                        (src_slice.read_unaligned().min(1f32).max(0f32) * 2048f32).round() as usize;

                    let dst = dst_ptr.add(px);
                    let transferred = *lut_table.get_unchecked(pixel.min(2048));

                    dst.write_unaligned(transferred);
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for (dst, src) in dst
            .chunks_exact_mut(dst_stride as usize)
            .zip(src_slice_safe_align.chunks_exact(src_stride as usize))
        {
            unsafe {
                let mut _cx = 0usize;

                let src_ptr = src.as_ptr() as *const f32;
                let dst_ptr = dst.as_mut_ptr();

                for x in _cx..width as usize {
                    let px = x;
                    let src_slice = src_ptr.add(px);
                    let pixel =
                        (src_slice.read_unaligned().min(1f32).max(0f32) * 2048f32).round() as usize;

                    let dst = dst_ptr.add(px);
                    let transferred = *lut_table.get_unchecked(pixel.min(2048));

                    dst.write_unaligned(transferred);
                }
            }
        }
    }
}

/// This function converts Linear to Plane. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear plane data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive Gamma plane data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_to_plane(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
