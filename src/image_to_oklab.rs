/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_image_to_oklab;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_image_to_oklab;
use crate::{Oklab, Rgb, TransferFunction};

#[inline(always)]
fn channels_to_oklab<const CHANNELS_CONFIGURATION: u8>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let channels = image_configuration.get_channels_count();

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    let mut _has_sse = false;

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    if is_x86_feature_detected!("sse4.1") {
        _has_sse = true;
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            _cx = neon_image_to_oklab::<CHANNELS_CONFIGURATION>(
                _cx,
                src.as_ptr(),
                src_offset,
                width,
                dst.as_mut_ptr(),
                dst_offset,
                transfer_function,
            )
        }

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        unsafe {
            if _has_sse {
                _cx = sse_image_to_oklab::<CHANNELS_CONFIGURATION>(
                    _cx,
                    src.as_ptr(),
                    src_offset,
                    width,
                    dst.as_mut_ptr(),
                    dst_offset,
                    transfer_function,
                )
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
            let oklab = Oklab::from_rgb(rgb, transfer_function);

            let dst_store = unsafe { dst_ptr.add(px) };

            unsafe {
                dst_store.write_unaligned(oklab.l);
                dst_store.add(1).write_unaligned(oklab.a);
                dst_store.add(2).write_unaligned(oklab.b);
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

/// This function converts RGB to Oklab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_oklab(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_oklab::<{ ImageConfiguration::Rgb as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts RGBA to Oklab against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_oklab(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_oklab::<{ ImageConfiguration::Rgba as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts BGRA to Oklab against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_oklab(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_oklab::<{ ImageConfiguration::Bgra as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts BGR to Oklab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_oklab(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_oklab::<{ ImageConfiguration::Bgr as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
