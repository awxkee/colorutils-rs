/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::image_to_jzazbz::JzazbzTarget;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_jzazbz_to_image;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_jzazbz_to_image;
use crate::{Jzazbz, Jzczhz, TransferFunction};

fn jzazbz_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let target: JzazbzTarget = TARGET.into();

    let mut _wide_row_handle: Option<
        unsafe fn(usize, *const f32, u32, *mut u8, u32, u32, f32, TransferFunction) -> usize,
    > = None;

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    if is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_jzazbz_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _wide_row_handle = Some(neon_jzazbz_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let channels = image_configuration.get_channels_count();

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *mut f32 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        if let Some(dispatcher) = _wide_row_handle {
            unsafe {
                _cx = dispatcher(
                    _cx,
                    src.as_ptr(),
                    src_offset as u32,
                    dst.as_mut_ptr(),
                    dst_offset as u32,
                    width,
                    display_luminance,
                    transfer_function,
                );
            }
        }

        for x in _cx..width as usize {
            let px = x * channels;
            let l_x = unsafe { src_ptr.add(px).read_unaligned() };
            let l_y = unsafe { src_ptr.add(px + 1).read_unaligned() };
            let l_z = unsafe { src_ptr.add(px + 2).read_unaligned() };
            let rgb;

            match target {
                JzazbzTarget::JZAZBZ => {
                    let jzazbz = Jzazbz::new_with_luminance(l_x, l_y, l_z, display_luminance);
                    rgb = jzazbz.to_rgb(transfer_function);
                }
                JzazbzTarget::JZCZHZ => {
                    let jzczhz = Jzczhz::new(l_x, l_y, l_z);
                    rgb = jzczhz.to_rgb_with_luminance(display_luminance, transfer_function);
                }
            }

            unsafe {
                let dst = dst_ptr.add(x * channels);
                dst.add(image_configuration.get_r_channel_offset())
                    .write_unaligned(rgb.r);
                dst.add(image_configuration.get_g_channel_offset())
                    .write_unaligned(rgb.g);
                dst.add(image_configuration.get_b_channel_offset())
                    .write_unaligned(rgb.b);
                if image_configuration.has_alpha() {
                    let l_a = src_ptr.add(px + 3).read_unaligned();
                    let a_value = (l_a * 255f32).max(0f32);
                    dst.add(image_configuration.get_a_channel_offset())
                        .write_unaligned(a_value as u8);
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts Jzazbz with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzazbz_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    jzazbz_to_image::<{ ImageConfiguration::Rgba as u8 }, { JzazbzTarget::JZAZBZ as u8 }>(
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

/// This function converts Jzazbz to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzazbz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzazbz_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    jzazbz_to_image::<{ ImageConfiguration::Rgb as u8 }, { JzazbzTarget::JZAZBZ as u8 }>(
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

/// This function converts Jzazbz to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzazbz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzazbz_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    jzazbz_to_image::<{ ImageConfiguration::Bgr as u8 }, { JzazbzTarget::JZAZBZ as u8 }>(
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

/// This function converts Jzazbz with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzazbz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzazbz_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    jzazbz_to_image::<{ ImageConfiguration::Bgra as u8 }, { JzazbzTarget::JZAZBZ as u8 }>(
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

/// This function converts Jzczhz with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzczhz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzczhz_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    jzazbz_to_image::<{ ImageConfiguration::Rgba as u8 }, { JzazbzTarget::JZCZHZ as u8 }>(
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

/// This function converts Jzczhz to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzczhz_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    jzazbz_to_image::<{ ImageConfiguration::Rgb as u8 }, { JzazbzTarget::JZCZHZ as u8 }>(
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

/// This function converts Jzczhz to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzczhz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzczhz_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    jzazbz_to_image::<{ ImageConfiguration::Bgr as u8 }, { JzazbzTarget::JZCZHZ as u8 }>(
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

/// This function converts Jzczhz with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzczhz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzczhz_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    display_luminance: f32,
    transfer_function: TransferFunction,
) {
    jzazbz_to_image::<{ ImageConfiguration::Bgra as u8 }, { JzazbzTarget::JZCZHZ as u8 }>(
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
