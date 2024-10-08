/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::planar_to_linear::neon_plane_to_linear;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_plane_to_linear;
use crate::TransferFunction;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
#[cfg(feature = "rayon")]
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
    let mut _wide_row_handler: Option<
        unsafe fn(usize, *const u8, usize, u32, *mut f32, usize, TransferFunction) -> usize,
    > = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler = Some(sse_plane_to_linear);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler = Some(neon_plane_to_linear);
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
            .for_each(|(dst, src)| unsafe {
                let mut _cx = 0usize;

                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr() as *mut f32;

                if let Some(dispatcher) = _wide_row_handler {
                    _cx = dispatcher(_cx, src_ptr, 0, width, dst_ptr, 0, transfer_function);
                }

                for x in _cx..width as usize {
                    let px = x;
                    let dst = dst_ptr.add(px);
                    let src = src_ptr.add(px);
                    let pixel_f = src.read_unaligned() as f32 * (1. / 255.);
                    let transferred = transfer_function.linearize(pixel_f);

                    dst.write_unaligned(transferred);
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

            if let Some(dispatcher) = _wide_row_handler {
                unsafe {
                    _cx = dispatcher(
                        _cx,
                        src.as_ptr(),
                        src_offset,
                        width,
                        dst.as_mut_ptr(),
                        dst_offset,
                        transfer_function,
                    );
                }
            }

            for x in _cx..width as usize {
                let px = x;
                let dst = unsafe { dst_ptr.add(px) };
                let src = unsafe { src_ptr.add(px) };
                let pixel_f = unsafe { src.read_unaligned() as f32 } * (1. / 255.);
                let transferred = transfer_function.linearize(pixel_f);

                unsafe {
                    dst.write_unaligned(transferred);
                }
            }

            src_offset += src_stride as usize;
            dst_offset += dst_stride as usize;
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
