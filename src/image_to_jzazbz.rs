/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_image_to_jzazbz;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_image_to_jzazbz;
use crate::{Jzazbz, Jzczhz, Rgb, TransferFunction, Xyz, SRGB_TO_XYZ_D65};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

#[repr(u8)]
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum JzazbzTarget {
    Jzazbz = 0,
    Jzczhz = 1,
}

impl From<u8> for JzazbzTarget {
    fn from(value: u8) -> Self {
        match value {
            0 => JzazbzTarget::Jzazbz,
            1 => JzazbzTarget::Jzczhz,
            _ => {
                panic!("Not known value {}", value)
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn channels_to_jzaz<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    let target: JzazbzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let channels = image_configuration.get_channels_count();

    let mut _wide_row_handle: Option<
        unsafe fn(usize, *const f32, usize, u32, *mut f32, usize, f32) -> usize,
    > = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handle = Some(neon_image_to_jzazbz::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_image_to_jzazbz::<CHANNELS_CONFIGURATION, TARGET>);
    }

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

                let dst_ptr = dst.as_mut_ptr() as *mut f32;

                let mut linearized_row = vec![0f32; width as usize * channels];
                for (linear_chunk, src_chunk) in linearized_row
                    .chunks_exact_mut(channels)
                    .zip(src.chunks_exact(channels))
                {
                    linear_chunk[image_configuration.get_r_channel_offset()] = *lut_table
                        .get_unchecked(
                            src_chunk[image_configuration.get_r_channel_offset()] as usize,
                        );
                    linear_chunk[image_configuration.get_g_channel_offset()] = *lut_table
                        .get_unchecked(
                            src_chunk[image_configuration.get_g_channel_offset()] as usize,
                        );
                    linear_chunk[image_configuration.get_b_channel_offset()] = *lut_table
                        .get_unchecked(
                            src_chunk[image_configuration.get_b_channel_offset()] as usize,
                        );
                    if image_configuration.has_alpha() {
                        linear_chunk[image_configuration.get_a_channel_offset()] =
                            src_chunk[image_configuration.get_a_channel_offset()] as f32
                                * (1. / 255.0);
                    }
                }

                if let Some(dispatcher) = _wide_row_handle {
                    _cx = dispatcher(
                        _cx,
                        linearized_row.as_ptr(),
                        0,
                        width,
                        dst.as_mut_ptr() as *mut f32,
                        0,
                        display_luminance,
                    );
                }

                for x in _cx..width as usize {
                    let px = x * channels;

                    let src = linearized_row.get_unchecked(px..);
                    let r = *src.get_unchecked(image_configuration.get_r_channel_offset());
                    let g = *src.get_unchecked(image_configuration.get_g_channel_offset());
                    let b = *src.get_unchecked(image_configuration.get_b_channel_offset());

                    let xyz = Xyz::from_linear_rgb(Rgb::<f32>::new(r, g, b), &SRGB_TO_XYZ_D65);

                    let dst_store = dst_ptr.add(px);

                    match target {
                        JzazbzTarget::Jzazbz => {
                            let jzazbz =
                                Jzazbz::from_xyz_with_display_luminance(xyz, display_luminance);

                            dst_store.write_unaligned(jzazbz.jz);
                            dst_store.add(1).write_unaligned(jzazbz.az);
                            dst_store.add(2).write_unaligned(jzazbz.bz);
                        }
                        JzazbzTarget::Jzczhz => {
                            let jzczhz =
                                Jzczhz::from_xyz_with_display_luminance(xyz, display_luminance);

                            dst_store.write_unaligned(jzczhz.jz);
                            dst_store.add(1).write_unaligned(jzczhz.cz);
                            dst_store.add(2).write_unaligned(jzczhz.hz);
                        }
                    }

                    if image_configuration.has_alpha() {
                        let a = *src.get_unchecked(image_configuration.get_a_channel_offset());
                        dst_store.add(3).write_unaligned(a);
                    }
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

                let dst_ptr = dst.as_mut_ptr() as *mut f32;

                let mut linearized_row = vec![0f32; width as usize * channels];
                for (linear_chunk, src_chunk) in linearized_row
                    .chunks_exact_mut(channels)
                    .zip(src.chunks_exact(channels))
                {
                    linear_chunk[image_configuration.get_r_channel_offset()] = *lut_table
                        .get_unchecked(
                            src_chunk[image_configuration.get_r_channel_offset()] as usize,
                        );
                    linear_chunk[image_configuration.get_g_channel_offset()] = *lut_table
                        .get_unchecked(
                            src_chunk[image_configuration.get_g_channel_offset()] as usize,
                        );
                    linear_chunk[image_configuration.get_b_channel_offset()] = *lut_table
                        .get_unchecked(
                            src_chunk[image_configuration.get_b_channel_offset()] as usize,
                        );
                    if image_configuration.has_alpha() {
                        linear_chunk[image_configuration.get_a_channel_offset()] =
                            src_chunk[image_configuration.get_a_channel_offset()] as f32
                                * (1. / 255.0);
                    }
                }

                if let Some(dispatcher) = _wide_row_handle {
                    _cx = dispatcher(
                        _cx,
                        linearized_row.as_ptr(),
                        0,
                        width,
                        dst.as_mut_ptr() as *mut f32,
                        0,
                        display_luminance,
                    );
                }

                for x in _cx..width as usize {
                    let px = x * channels;

                    let src = linearized_row.get_unchecked(px..);
                    let r = *src.get_unchecked(image_configuration.get_r_channel_offset());
                    let g = *src.get_unchecked(image_configuration.get_g_channel_offset());
                    let b = *src.get_unchecked(image_configuration.get_b_channel_offset());

                    let xyz = Xyz::from_linear_rgb(Rgb::<f32>::new(r, g, b), &SRGB_TO_XYZ_D65);

                    let dst_store = dst_ptr.add(px);

                    match target {
                        JzazbzTarget::Jzazbz => {
                            let jzazbz =
                                Jzazbz::from_xyz_with_display_luminance(xyz, display_luminance);

                            dst_store.write_unaligned(jzazbz.jz);
                            dst_store.add(1).write_unaligned(jzazbz.az);
                            dst_store.add(2).write_unaligned(jzazbz.bz);
                        }
                        JzazbzTarget::Jzczhz => {
                            let jzczhz =
                                Jzczhz::from_xyz_with_display_luminance(xyz, display_luminance);

                            dst_store.write_unaligned(jzczhz.jz);
                            dst_store.add(1).write_unaligned(jzczhz.cz);
                            dst_store.add(2).write_unaligned(jzczhz.hz);
                        }
                    }

                    if image_configuration.has_alpha() {
                        let a = *src.get_unchecked(image_configuration.get_a_channel_offset());
                        dst_store.add(3).write_unaligned(a);
                    }
                }
            }
        }
    }
}

/// This function converts RGB to Jzazbz against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzazbz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_jzazbz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    channels_to_jzaz::<{ ImageConfiguration::Rgb as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        display_luminance,
        transfer_function,
    );
}

/// This function converts RGBA to Jzazbz against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzazbz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_jzazbz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    channels_to_jzaz::<{ ImageConfiguration::Rgba as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        display_luminance,
        transfer_function,
    );
}

/// This function converts BGRA to Jzazbz against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzazbz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_jzazbz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    channels_to_jzaz::<{ ImageConfiguration::Bgra as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        display_luminance,
        transfer_function,
    );
}

/// This function converts BGR to Jzazbz against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzazbz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_jzazbz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    channels_to_jzaz::<{ ImageConfiguration::Bgr as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        display_luminance,
        transfer_function,
    );
}

/// This function converts RGB to Jzczhz against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzczhz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_jzczhz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    channels_to_jzaz::<{ ImageConfiguration::Rgb as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        display_luminance,
        transfer_function,
    );
}

/// This function converts RGBA to Jzczhz against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzczhz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_jzczhz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    channels_to_jzaz::<{ ImageConfiguration::Rgba as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        display_luminance,
        transfer_function,
    );
}

/// This function converts BGRA to Jzczhz against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzczhz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_jzczhz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    channels_to_jzaz::<{ ImageConfiguration::Bgra as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        display_luminance,
        transfer_function,
    );
}

/// This function converts BGR to Jzczhz against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzczhz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_jzczhz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    channels_to_jzaz::<{ ImageConfiguration::Bgr as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        display_luminance,
        transfer_function,
    );
}
