/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_oklab_to_image;
use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_oklab_to_image;
use crate::oklch::Oklch;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_oklab_to_image;
use crate::{Oklab, Rgb, TransferFunction};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

#[allow(clippy::type_complexity)]
fn oklab_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let target: OklabTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let mut _wide_row_handle: Option<
        unsafe fn(usize, *const f32, usize, *mut f32, u32, u32) -> usize,
    > = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_oklab_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if is_x86_feature_detected!("avx2") {
        _wide_row_handle = Some(avx_oklab_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handle = Some(neon_oklab_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let mut lut_table = vec![0u8; 2049];
    for i in 0..2049 {
        lut_table[i] = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.)
            .ceil()
            .min(255.) as u8;
    }

    let channels = image_configuration.get_channels_count();

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

                let mut transient_row = vec![0f32; width as usize * channels];

                let src_ptr = src.as_ptr() as *mut f32;

                if let Some(dispatcher) = _wide_row_handle {
                    _cx = dispatcher(_cx, src_ptr, 0, transient_row.as_mut_ptr(), 0, width)
                }

                for x in _cx..width as usize {
                    let px = x * channels;
                    let source_p = src_ptr.add(px);
                    let l_x = source_p.read_unaligned();
                    let l_y = source_p.add(1).read_unaligned();
                    let l_z = source_p.add(2).read_unaligned();
                    let rgb = match target {
                        OklabTarget::Oklab => {
                            let oklab = Oklab::new(l_x, l_y, l_z);
                            oklab.to_linear_rgb()
                        }
                        OklabTarget::Oklch => {
                            let oklch = Oklch::new(l_x, l_y, l_z);
                            oklch.to_linear_rgb()
                        }
                    };

                    let v_dst = transient_row.get_unchecked_mut((x * channels)..);
                    *v_dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) = rgb.r;
                    *v_dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) = rgb.g;
                    *v_dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) = rgb.b;
                    if image_configuration.has_alpha() {
                        let l_a = source_p.add(3).read_unaligned();
                        *v_dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) = l_a;
                    }
                }

                for (dst_chunks, src_chunks) in dst
                    .chunks_exact_mut(channels)
                    .zip(transient_row.chunks_exact_mut(channels))
                {
                    let rgb = (Rgb::<f32>::new(src_chunks[0], src_chunks[1], src_chunks[2])
                        * Rgb::<f32>::dup(2048f32))
                    .cast::<u16>();

                    dst_chunks[0] = *lut_table.get_unchecked(rgb.r as usize);
                    dst_chunks[1] = *lut_table.get_unchecked(rgb.g as usize);
                    dst_chunks[2] = *lut_table.get_unchecked(rgb.b as usize);
                    if image_configuration.has_alpha() {
                        let a_lin = (src_chunks[4] * 255f32).round() as u8;
                        dst_chunks[0] = a_lin;
                    }
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for (dst, src) in dst.chunks_exact_mut(dst_stride as usize)
            .zip(src_slice_safe_align.chunks_exact(src_stride as usize)) {
            unsafe {
                let mut _cx = 0usize;

                let mut transient_row = vec![0f32; width as usize * channels];

                let src_ptr = src.as_ptr() as *mut f32;

                if let Some(dispatcher) = _wide_row_handle {
                    _cx = dispatcher(_cx, src_ptr, 0, transient_row.as_mut_ptr(), 0, width)
                }

                for x in _cx..width as usize {
                    let px = x * channels;
                    let source_p = src_ptr.add(px);
                    let l_x = source_p.read_unaligned();
                    let l_y = source_p.add(1).read_unaligned();
                    let l_z = source_p.add(2).read_unaligned();
                    let rgb = match target {
                        OklabTarget::Oklab => {
                            let oklab = Oklab::new(l_x, l_y, l_z);
                            oklab.to_linear_rgb()
                        }
                        OklabTarget::Oklch => {
                            let oklch = Oklch::new(l_x, l_y, l_z);
                            oklch.to_linear_rgb()
                        }
                    };

                    let v_dst = transient_row.get_unchecked_mut((x * channels)..);
                    *v_dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) = rgb.r;
                    *v_dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) = rgb.g;
                    *v_dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) = rgb.b;
                    if image_configuration.has_alpha() {
                        let l_a = source_p.add(3).read_unaligned();
                        *v_dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) = l_a;
                    }
                }

                for (dst_chunks, src_chunks) in dst
                    .chunks_exact_mut(channels)
                    .zip(transient_row.chunks_exact_mut(channels))
                {
                    let rgb = (Rgb::<f32>::new(src_chunks[0], src_chunks[1], src_chunks[2])
                        * Rgb::<f32>::dup(2048f32))
                        .cast::<u16>();

                    dst_chunks[0] = *lut_table.get_unchecked(rgb.r as usize);
                    dst_chunks[1] = *lut_table.get_unchecked(rgb.g as usize);
                    dst_chunks[2] = *lut_table.get_unchecked(rgb.b as usize);
                    if image_configuration.has_alpha() {
                        let a_lin = (src_chunks[4] * 255f32).round() as u8;
                        dst_chunks[0] = a_lin;
                    }
                }
            }
        }
    }
}

/// This function converts Oklab with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklab_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    oklab_to_image::<{ ImageConfiguration::Rgba as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Oklab to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklab_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    oklab_to_image::<{ ImageConfiguration::Rgb as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Oklab to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklab_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    oklab_to_image::<{ ImageConfiguration::Bgr as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Oklab with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklab_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    oklab_to_image::<{ ImageConfiguration::Bgra as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts *Oklch* with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCH data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklch_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    oklab_to_image::<{ ImageConfiguration::Rgba as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts *Oklch* to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCH data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklch_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    oklab_to_image::<{ ImageConfiguration::Rgb as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts *Oklch* to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCH data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklch_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    oklab_to_image::<{ ImageConfiguration::Bgr as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts *Oklch* with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCH data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklch_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    oklab_to_image::<{ ImageConfiguration::Bgra as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
