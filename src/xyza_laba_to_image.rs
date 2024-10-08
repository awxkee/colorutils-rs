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
use crate::{LCh, Lab, Luv, Xyz, XYZ_TO_SRGB_D65};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
#[cfg(feature = "rayon")]
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
        unsafe fn(
            usize,
            *const f32,
            usize,
            *mut u8,
            usize,
            u32,
            &[[f32; 3]; 3],
            TransferFunction,
        ) -> usize,
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

    #[cfg(feature = "rayon")]
    {
        let src_slice_safe_align = unsafe {
            slice::from_raw_parts(
                src.as_ptr() as *const u8,
                src_stride as usize * height as usize,
            )
        };
        dst.par_chunks_exact_mut(dst_stride as usize)
            .zip(src_slice_safe_align.par_chunks_exact(src_stride as usize))
            .for_each(|(dst, src)| unsafe {
                let channels = image_configuration.get_channels_count();

                let mut _cx = 0usize;

                if let Some(dispatcher) = _wide_row_handler {
                    _cx = dispatcher(
                        _cx,
                        src.as_ptr() as *const f32,
                        0,
                        dst.as_mut_ptr(),
                        0,
                        width,
                        matrix,
                        transfer_function,
                    )
                }

                let src_ptr = src.as_ptr() as *mut f32;
                let dst_ptr = dst.as_mut_ptr();

                for x in _cx..width as usize {
                    let px = x * 4;
                    let l_x = src_ptr.add(px).read_unaligned();
                    let l_y = src_ptr.add(px + 1).read_unaligned();
                    let l_z = src_ptr.add(px + 2).read_unaligned();
                    let rgb = match source {
                        XyzTarget::Lab => {
                            let lab = Lab::new(l_x, l_y, l_z);
                            lab.to_rgb()
                        }
                        XyzTarget::Xyz => {
                            let xyz = Xyz::new(l_x, l_y, l_z);
                            xyz.to_rgb(matrix, transfer_function)
                        }
                        XyzTarget::Luv => {
                            let luv = Luv::new(l_x, l_y, l_z);
                            luv.to_rgb()
                        }
                        XyzTarget::Lch => {
                            let lch = LCh::new(l_x, l_y, l_z);
                            lch.to_rgb()
                        }
                    };

                    let l_a = src_ptr.add(px + 3).read_unaligned();
                    let a_value = (l_a * 255f32).max(0f32);
                    let dst = dst_ptr.add(x * channels);
                    dst.add(image_configuration.get_r_channel_offset())
                        .write_unaligned(rgb.r);
                    dst.add(image_configuration.get_g_channel_offset())
                        .write_unaligned(rgb.g);
                    dst.add(image_configuration.get_b_channel_offset())
                        .write_unaligned(rgb.b);
                    dst.add(image_configuration.get_a_channel_offset())
                        .write_unaligned(a_value as u8);
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;

        let channels = image_configuration.get_channels_count();

        for _ in 0..height as usize {
            let mut _cx = 0usize;

            if let Some(dispatcher) = _wide_row_handler {
                unsafe {
                    _cx = dispatcher(
                        _cx,
                        src.as_ptr(),
                        src_offset,
                        dst.as_mut_ptr(),
                        dst_offset,
                        width,
                        matrix,
                        transfer_function,
                    )
                }
            }

            let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *mut f32 };
            let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

            for x in _cx..width as usize {
                let px = x * 4;
                let l_x = unsafe { src_ptr.add(px).read_unaligned() };
                let l_y = unsafe { src_ptr.add(px + 1).read_unaligned() };
                let l_z = unsafe { src_ptr.add(px + 2).read_unaligned() };
                let rgb = match source {
                    XyzTarget::Lab => {
                        let lab = Lab::new(l_x, l_y, l_z);
                        lab.to_rgb()
                    }
                    XyzTarget::Xyz => {
                        let xyz = Xyz::new(l_x, l_y, l_z);
                        xyz.to_rgb(matrix, transfer_function)
                    }
                    XyzTarget::Luv => {
                        let luv = Luv::new(l_x, l_y, l_z);
                        luv.to_rgb()
                    }
                    XyzTarget::Lch => {
                        let lch = LCh::new(l_x, l_y, l_z);
                        lch.to_rgb()
                    }
                };

                let l_a = unsafe { src_ptr.add(px + 3).read_unaligned() };
                let a_value = (l_a * 255f32).max(0f32);
                unsafe {
                    let dst = dst_ptr.add(x * channels);
                    dst.add(image_configuration.get_r_channel_offset())
                        .write_unaligned(rgb.r);
                    dst.add(image_configuration.get_g_channel_offset())
                        .write_unaligned(rgb.g);
                    dst.add(image_configuration.get_b_channel_offset())
                        .write_unaligned(rgb.b);
                    dst.add(image_configuration.get_a_channel_offset())
                        .write_unaligned(a_value as u8);
                }
            }

            src_offset += src_stride as usize;
            dst_offset += dst_stride as usize;
        }
    }
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
pub fn lab_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
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
pub fn lab_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
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
pub fn luv_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
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
pub fn luv_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
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
pub fn xyz_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
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
pub fn xyz_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
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
pub fn lch_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
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
pub fn lch_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}
