/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_xyza_to_image;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_xyza_to_image;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_xyza_to_image;
use crate::xyz_target::XyzTarget;
use crate::{LCh, Lab, Luv, Xyz};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

#[allow(clippy::type_complexity)]
fn xyz_with_alpha_to_channels<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let source: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut _wide_row_handler: Option<
        unsafe fn(usize, *const f32, usize, *mut f32, usize, u32, &[[f32; 3]; 3]) -> usize,
    > = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler = Some(neon_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler = Some(sse_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handler = Some(avx_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

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
        let channels = image_configuration.get_channels_count();

        let mut _cx = 0usize;

        let mut transient_row = vec![0f32; width as usize * channels];

        if let Some(dispatcher) = _wide_row_handler {
            _cx = dispatcher(
                _cx,
                src.as_ptr() as *const f32,
                0,
                transient_row.as_mut_ptr(),
                0,
                width,
                matrix,
            )
        }

        let src_ptr = src.as_ptr() as *mut f32;

        for x in _cx..width as usize {
            let px = x * 4;
            let l_x = src_ptr.add(px).read_unaligned();
            let l_y = src_ptr.add(px + 1).read_unaligned();
            let l_z = src_ptr.add(px + 2).read_unaligned();
            let rgb = match source {
                XyzTarget::Lab => {
                    let lab = Lab::new(l_x, l_y, l_z);
                    lab.to_linear_rgb(matrix)
                }
                XyzTarget::Xyz => {
                    let xyz = Xyz::new(l_x, l_y, l_z);
                    xyz.to_linear_rgb(matrix)
                }
                XyzTarget::Luv => {
                    let luv = Luv::new(l_x, l_y, l_z);
                    luv.to_linear_rgb(matrix)
                }
                XyzTarget::Lch => {
                    let lch = LCh::new(l_x, l_y, l_z);
                    lch.to_linear_rgb(matrix)
                }
            };

            let l_a = src_ptr.add(px + 3).read_unaligned();
            let dst = transient_row.get_unchecked_mut((x * channels)..);
            *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) = rgb.r;
            *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) = rgb.g;
            *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) = rgb.b;
            *dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) = l_a;
        }

        for (dst_chunk, src_chunks) in dst
            .chunks_exact_mut(channels)
            .zip(transient_row.chunks_exact(channels))
        {
            let r_cast = (src_chunks[image_configuration.get_r_channel_offset()]
                .min(1.)
                .max(0.)
                * 2048f32)
                .round();
            let g_cast = (src_chunks[image_configuration.get_g_channel_offset()]
                .min(1.)
                .max(0.)
                * 2048f32)
                .round();
            let b_cast = (src_chunks[image_configuration.get_b_channel_offset()]
                .min(1.)
                .max(0.)
                * 2048f32)
                .round();
            let a_cast = (src_chunks[image_configuration.get_a_channel_offset()] * 255.)
                .min(255.)
                .max(0.) as u8;

            dst_chunk[image_configuration.get_r_channel_offset()] =
                *lut_table.get_unchecked(r_cast as usize);
            dst_chunk[image_configuration.get_g_channel_offset()] =
                *lut_table.get_unchecked(g_cast as usize);
            dst_chunk[image_configuration.get_b_channel_offset()] =
                *lut_table.get_unchecked(b_cast as usize);
            dst_chunk[image_configuration.get_a_channel_offset()] = a_cast;
        }
    });
}

/// This function converts LAB with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn lab_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lab as u8 }>(
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

/// This function converts LAB with separate alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn lab_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
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

/// This function converts LUV with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LUV data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn luv_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Luv as u8 }>(
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

/// This function converts LUV with separate alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LUV data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn luv_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
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

/// This function converts XYZ with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains XYZa data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn xyz_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Xyz as u8 }>(
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

/// This function converts XYZ with separate alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains XYZ data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn xyz_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Xyz as u8 }>(
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

/// This function converts LCH with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCHa data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn lch_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lch as u8 }>(
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

/// This function converts LCH with separate alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCHa data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn lch_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lch as u8 }>(
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
