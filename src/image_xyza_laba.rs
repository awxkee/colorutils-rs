/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_channels_to_xyza_or_laba;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_channels_to_xyza_laba;
use crate::xyz_target::XyzTarget;
use crate::{LCh, Lab, Luv, Rgb, TransferFunction, Xyz};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

#[allow(clippy::type_complexity)]
fn channels_to_xyz_with_alpha<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut _wide_row_handler: Option<
        unsafe fn(usize, *const f32, usize, u32, *mut f32, usize, &[[f32; 3]; 3]) -> usize,
    > = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _wide_row_handler = Some(sse_channels_to_xyza_laba::<CHANNELS_CONFIGURATION, TARGET>);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler = Some(neon_channels_to_xyza_or_laba::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let channels = image_configuration.get_channels_count();

    let mut lut_table = vec![0f32; 256];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = transfer_function.linearize(i as f32 * (1. / 255.0));
    }

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

        let mut transient_row = vec![0f32; width as usize * channels];

        for (dst_chunk, src_chunks) in transient_row
            .chunks_exact_mut(channels)
            .zip(src.chunks_exact(channels))
        {
            dst_chunk[image_configuration.get_r_channel_offset()] = *lut_table
                .get_unchecked(src_chunks[image_configuration.get_r_channel_offset()] as usize);
            dst_chunk[image_configuration.get_g_channel_offset()] = *lut_table
                .get_unchecked(src_chunks[image_configuration.get_g_channel_offset()] as usize);
            dst_chunk[image_configuration.get_b_channel_offset()] = *lut_table
                .get_unchecked(src_chunks[image_configuration.get_b_channel_offset()] as usize);
            dst_chunk[image_configuration.get_a_channel_offset()] =
                src_chunks[image_configuration.get_a_channel_offset()] as f32 * (1. / 255.0);
        }

        if let Some(dispatcher) = _wide_row_handler {
            _cx = dispatcher(
                _cx,
                transient_row.as_ptr(),
                0,
                width,
                dst.as_mut_ptr() as *mut f32,
                0,
                matrix,
            );
        }

        let dst_ptr = dst.as_mut_ptr() as *mut f32;

        for x in _cx..width as usize {
            let px = x * channels;
            let src = transient_row.get_unchecked(px..);

            let r = *src.get_unchecked(image_configuration.get_r_channel_offset());
            let g = *src.get_unchecked(image_configuration.get_g_channel_offset());
            let b = *src.get_unchecked(image_configuration.get_b_channel_offset());

            let rgb = Rgb::<f32>::new(r, g, b);
            let px = x * channels;
            let dst_store = dst_ptr.add(px);

            let xyz = Xyz::from_linear_rgb(rgb, matrix);

            match target {
                XyzTarget::Lab => {
                    let lab = Lab::from_xyz(xyz);
                    dst_store.write_unaligned(lab.l);
                    dst_store.add(1).write_unaligned(lab.a);
                    dst_store.add(2).write_unaligned(lab.b);
                }
                XyzTarget::Xyz => {
                    dst_store.write_unaligned(xyz.x);
                    dst_store.add(1).write_unaligned(xyz.y);
                    dst_store.add(2).write_unaligned(xyz.z);
                }
                XyzTarget::Luv => {
                    let luv = Luv::from_xyz(xyz);
                    dst_store.write_unaligned(luv.l);
                    dst_store.add(1).write_unaligned(luv.u);
                    dst_store.add(2).write_unaligned(luv.v);
                }
                XyzTarget::Lch => {
                    let luv = Luv::from_xyz(xyz);
                    let lch = LCh::from_luv(luv);
                    dst_store.write_unaligned(lch.l);
                    dst_store.add(1).write_unaligned(lch.c);
                    dst_store.add(2).write_unaligned(lch.h);
                }
            }
            let a = *src.get_unchecked(image_configuration.get_a_channel_offset());
            dst_store.add(3).write_unaligned(a);
        }
    });
}

/// This function converts RGBA to CIE L*ab.
///
/// This function converts RGBA to CIE L*ab against D65 white point and preserving
/// and normalizing alpha channels keeping it at last positions.
/// This is much more effective than naive direct transformation
///
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn rgba_to_lab_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts BGRA to CIE L*ab.
///
/// This function converts BGRA to CIE L*ab against D65 white point
/// and preserving and normalizing alpha channels keeping it at last positions.
/// This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn bgra_to_lab_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts RGBA to CIE L*uv.
///
/// This function converts RGBA to CIE L*uv against D65 white point and preserving
/// and normalizing alpha channels keeping it at last positions.
/// This is much more effective than naive direct transformation.
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn rgba_to_luv_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts BGRA to CIE L*uv.
///
/// This function converts BGRA to CIE L*uv against D65 white point
/// and preserving and normalizing alpha channels keeping it at last positions.
/// This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn bgra_to_luv_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts RGBA to CIE XYZ against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn rgba_to_xyz_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts BGRA to CIE XYZ against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn bgra_to_xyz_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts RGBA to CIE LCH against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn rgba_to_lch_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts BGRA to CIE LCH against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn bgra_to_lch_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}
