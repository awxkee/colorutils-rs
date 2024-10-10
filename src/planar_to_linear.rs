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

#[inline(always)]
#[allow(clippy::type_complexity)]
fn channels_to_linear(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let dst_slice_safe_align = unsafe {
        slice::from_raw_parts_mut(
            dst.as_mut_ptr() as *mut u8,
            dst_stride as usize * height as usize,
        )
    };

    let mut lut_table = vec![0f32; 256];
    for i in 0..256 {
        lut_table[i] = transfer_function.linearize(i as f32 * (1. / 255.0));
    }

    #[cfg(feature = "rayon")]
    {
        dst_slice_safe_align
            .par_chunks_exact_mut(dst_stride as usize)
            .zip(src.par_chunks_exact(src_stride as usize))
            .for_each(|(dst, src)| unsafe {
                let mut _cx = 0usize;

                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr() as *mut f32;

                for x in _cx..width as usize {
                    let px = x;
                    let dst = dst_ptr.add(px);
                    let src = src_ptr.add(px);
                    let transferred = *lut_table.get_unchecked(src.read_unaligned() as usize);

                    dst.write_unaligned(transferred);
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for (dst, src) in dst_slice_safe_align
            .chunks_exact_mut(dst_stride as usize)
            .zip(src.chunks_exact(src_stride as usize))
        {
            unsafe {
                let mut _cx = 0usize;

                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr() as *mut f32;

                for x in _cx..width as usize {
                    let px = x;
                    let dst = dst_ptr.add(px);
                    let src = src_ptr.add(px);
                    let transferred = *lut_table.get_unchecked(src.read_unaligned() as usize);

                    dst.write_unaligned(transferred);
                }
            }
        }
    }
}

/// This function converts Plane to Linear. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn plane_to_linear(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
