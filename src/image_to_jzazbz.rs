/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::image_to_jzazbz::JzazbzTarget::{JZAZBZ, JZCZHZ};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_image_to_jzazbz;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_image_to_jzazbz;
use crate::{Jzazbz, Jzczhz, Rgb, TransferFunction};

#[repr(u8)]
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum JzazbzTarget {
    JZAZBZ = 0,
    JZCZHZ = 1,
}

impl From<u8> for JzazbzTarget {
    fn from(value: u8) -> Self {
        match value {
            0 => JZAZBZ,
            1 => JZCZHZ,
            _ => {
                panic!("Not known value {}", value)
            }
        }
    }
}

#[inline(always)]
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
        unsafe fn(usize, *const u8, usize, u32, *mut f32, usize, f32, TransferFunction) -> usize,
    > = None;

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _wide_row_handle = Some(neon_image_to_jzazbz::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    if is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_image_to_jzazbz::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

        if let Some(dispatcher) = _wide_row_handle {
            unsafe {
                _cx = dispatcher(
                    _cx,
                    src.as_ptr(),
                    src_offset,
                    width,
                    dst.as_mut_ptr(),
                    dst_offset,
                    display_luminance,
                    transfer_function,
                );
            }
        }

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

            let dst_store = unsafe { dst_ptr.add(px) };

            match target {
                JZAZBZ => {
                    let jzazbz =
                        Jzazbz::from_rgb_with_luminance(rgb, display_luminance, transfer_function);
                    unsafe {
                        dst_store.write_unaligned(jzazbz.jz);
                        dst_store.add(1).write_unaligned(jzazbz.az);
                        dst_store.add(2).write_unaligned(jzazbz.bz);
                    }
                }
                JZCZHZ => {
                    let jzczhz =
                        Jzczhz::from_rgb_with_luminance(rgb, display_luminance, transfer_function);
                    unsafe {
                        dst_store.write_unaligned(jzczhz.jz);
                        dst_store.add(1).write_unaligned(jzczhz.cz);
                        dst_store.add(2).write_unaligned(jzczhz.hz);
                    }
                }
            }

            if image_configuration.has_alpha() {
                let a = unsafe {
                    src.add(image_configuration.get_a_channel_offset())
                        .read_unaligned()
                };
                let a_lin = a as f32 * (1f32 / 255f32);
                unsafe {
                    dst_store.add(3).write_unaligned(a_lin);
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
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
    channels_to_jzaz::<{ ImageConfiguration::Rgb as u8 }, { JZAZBZ as u8 }>(
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
    channels_to_jzaz::<{ ImageConfiguration::Rgba as u8 }, { JZAZBZ as u8 }>(
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
    channels_to_jzaz::<{ ImageConfiguration::Bgra as u8 }, { JZAZBZ as u8 }>(
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
    channels_to_jzaz::<{ ImageConfiguration::Bgr as u8 }, { JZAZBZ as u8 }>(
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
    channels_to_jzaz::<{ ImageConfiguration::Rgb as u8 }, { JZCZHZ as u8 }>(
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
    channels_to_jzaz::<{ ImageConfiguration::Rgba as u8 }, { JZCZHZ as u8 }>(
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
    channels_to_jzaz::<{ ImageConfiguration::Bgra as u8 }, { JZCZHZ as u8 }>(
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
    channels_to_jzaz::<{ ImageConfiguration::Bgr as u8 }, { JZCZHZ as u8 }>(
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
