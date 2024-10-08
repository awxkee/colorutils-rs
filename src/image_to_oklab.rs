/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_image_to_oklab;
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_image_to_oklab;
use crate::oklch::Oklch;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_image_to_oklab;
use crate::{Oklab, Rgb, TransferFunction};

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum OklabTarget {
    Oklab = 0,
    Oklch = 1,
}

impl From<u8> for OklabTarget {
    fn from(value: u8) -> Self {
        match value {
            0 => OklabTarget::Oklab,
            1 => OklabTarget::Oklch,
            _ => {
                panic!("Not implemented")
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn channels_to_oklab<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let target: OklabTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let channels = image_configuration.get_channels_count();

    let mut _wide_row_handle: Option<
        unsafe fn(usize, *const u8, usize, u32, *mut f32, usize, TransferFunction) -> usize,
    > = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handle = Some(neon_image_to_oklab::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_image_to_oklab::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handle = Some(avx_image_to_oklab::<CHANNELS_CONFIGURATION, TARGET>);
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
            let dst_store = unsafe { dst_ptr.add(px) };

            match target {
                OklabTarget::Oklab => {
                    let oklab = Oklab::from_rgb(rgb, transfer_function);
                    unsafe {
                        dst_store.write_unaligned(oklab.l);
                        dst_store.add(1).write_unaligned(oklab.a);
                        dst_store.add(2).write_unaligned(oklab.b);
                    }
                }
                OklabTarget::Oklch => {
                    let oklch = Oklch::from_rgb(rgb, transfer_function);
                    unsafe {
                        dst_store.write_unaligned(oklch.l);
                        dst_store.add(1).write_unaligned(oklch.c);
                        dst_store.add(2).write_unaligned(oklch.h);
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
    channels_to_oklab::<{ ImageConfiguration::Rgb as u8 }, { OklabTarget::Oklab as u8 }>(
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
    channels_to_oklab::<{ ImageConfiguration::Rgba as u8 }, { OklabTarget::Oklab as u8 }>(
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
    channels_to_oklab::<{ ImageConfiguration::Bgra as u8 }, { OklabTarget::Oklab as u8 }>(
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
    channels_to_oklab::<{ ImageConfiguration::Bgr as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts RGB to Oklch against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_oklch(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_oklab::<{ ImageConfiguration::Rgb as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts RGBA to Oklch against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_oklch(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_oklab::<{ ImageConfiguration::Rgba as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts BGRA to Oklch against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_oklch(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_oklab::<{ ImageConfiguration::Bgra as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts BGR to Oklch against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_oklch(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_oklab::<{ ImageConfiguration::Bgr as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
