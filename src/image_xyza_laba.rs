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
use crate::{Rgb, TransferFunction, Xyz, SRGB_TO_XYZ_D65};

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

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let mut _wide_row_handler: Option<
        unsafe fn(
            usize,
            *const u8,
            usize,
            u32,
            *mut f32,
            usize,
            &[[f32; 3]; 3],
            TransferFunction,
        ) -> usize,
    > = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _wide_row_handler = match transfer_function {
                TransferFunction::Srgb => Some(
                    sse_channels_to_xyza_laba::<
                        CHANNELS_CONFIGURATION,
                        TARGET,
                        { TransferFunction::Srgb as u8 },
                    >,
                ),
                TransferFunction::Rec709 => Some(
                    sse_channels_to_xyza_laba::<
                        CHANNELS_CONFIGURATION,
                        TARGET,
                        { TransferFunction::Rec709 as u8 },
                    >,
                ),
                TransferFunction::Gamma2p2 => Some(
                    sse_channels_to_xyza_laba::<
                        CHANNELS_CONFIGURATION,
                        TARGET,
                        { TransferFunction::Gamma2p2 as u8 },
                    >,
                ),
                TransferFunction::Gamma2p8 => Some(
                    sse_channels_to_xyza_laba::<
                        CHANNELS_CONFIGURATION,
                        TARGET,
                        { TransferFunction::Gamma2p8 as u8 },
                    >,
                ),
            };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler = match transfer_function {
            TransferFunction::Srgb => Some(
                neon_channels_to_xyza_or_laba::<
                    CHANNELS_CONFIGURATION,
                    TARGET,
                    { TransferFunction::Srgb as u8 },
                >,
            ),
            TransferFunction::Rec709 => Some(
                neon_channels_to_xyza_or_laba::<
                    CHANNELS_CONFIGURATION,
                    TARGET,
                    { TransferFunction::Rec709 as u8 },
                >,
            ),
            TransferFunction::Gamma2p2 => Some(
                neon_channels_to_xyza_or_laba::<
                    CHANNELS_CONFIGURATION,
                    TARGET,
                    { TransferFunction::Gamma2p2 as u8 },
                >,
            ),
            TransferFunction::Gamma2p8 => Some(
                neon_channels_to_xyza_or_laba::<
                    CHANNELS_CONFIGURATION,
                    TARGET,
                    { TransferFunction::Gamma2p8 as u8 },
                >,
            ),
        };
    }

    let channels = image_configuration.get_channels_count();

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        if let Some(dispatcher) = _wide_row_handler {
            unsafe {
                _cx = dispatcher(
                    _cx,
                    src.as_ptr(),
                    src_offset,
                    width,
                    dst.as_mut_ptr(),
                    dst_offset,
                    matrix,
                    transfer_function,
                );
            }
        }

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

        for x in _cx..width as usize {
            let px = x * channels;
            let src = unsafe { src_ptr.add(px) };
            let r = unsafe {
                src.add(image_configuration.get_r_channel_offset())
                    .read_unaligned()
            };
            let g = unsafe {
                src.add(image_configuration.get_g_channel_offset())
                    .read_unaligned()
            };
            let b = unsafe {
                src.add(image_configuration.get_b_channel_offset())
                    .read_unaligned()
            };

            let rgb = Rgb::<u8>::new(r, g, b);
            let px = x * channels;
            let dst_store = unsafe { dst_ptr.add(px) };
            match target {
                XyzTarget::Lab => {
                    let lab = rgb.to_lab();
                    unsafe {
                        dst_store.write_unaligned(lab.l);
                        dst_store.add(1).write_unaligned(lab.a);
                        dst_store.add(2).write_unaligned(lab.b);
                    }
                }
                XyzTarget::Xyz => {
                    let xyz = Xyz::from_rgb(rgb, matrix, transfer_function);
                    unsafe {
                        dst_store.write_unaligned(xyz.x);
                        dst_store.add(1).write_unaligned(xyz.y);
                        dst_store.add(2).write_unaligned(xyz.z);
                    }
                }
                XyzTarget::Luv => {
                    let luv = rgb.to_luv();
                    unsafe {
                        dst_store.write_unaligned(luv.l);
                        dst_store.add(1).write_unaligned(luv.u);
                        dst_store.add(2).write_unaligned(luv.v);
                    }
                }
                XyzTarget::Lch => {
                    let lch = rgb.to_lch();
                    unsafe {
                        dst_store.write_unaligned(lch.l);
                        dst_store.add(1).write_unaligned(lch.c);
                        dst_store.add(2).write_unaligned(lch.h);
                    }
                }
            }

            let a = unsafe {
                src.add(image_configuration.get_a_channel_offset())
                    .read_unaligned()
            };
            let a_lin = a as f32 * (1f32 / 255f32);
            unsafe {
                dst_store.add(3).write_unaligned(a_lin);
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
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
pub fn rgba_to_lab_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
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
pub fn bgra_to_lab_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
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
pub fn rgba_to_luv_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
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
pub fn bgra_to_luv_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
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
pub fn rgba_to_xyz_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
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
pub fn bgra_to_xyz_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
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
pub fn rgba_to_lch_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
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
pub fn bgra_to_lch_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}
