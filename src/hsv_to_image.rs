/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::slice;

use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_hsv_u16_to_image;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_hsv_u16_to_image;
use crate::{Hsl, Hsv};

#[inline(always)]
fn hsv_u16_to_channels<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: &[u16],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    let target: HsvTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Alpha may be set only on images with alpha");
        }
    }

    let mut _wide_row_handler: Option<
        unsafe fn(usize, *const u16, usize, u32, *mut u8, usize, f32) -> usize,
    > = None;

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    if is_x86_feature_detected!("sse4.1") {
        _wide_row_handler = Some(sse_hsv_u16_to_image::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "arm"),
        target_feature = "neon"
    ))]
    {
        _wide_row_handler =
            Some(neon_hsv_u16_to_image::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let channels = image_configuration.get_channels_count();

    let scale = 1f32 / scale;

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
                    scale,
                );
            }
        }

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *const u16 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

        for x in _cx..width as usize {
            let px = x * channels;
            let src = unsafe { src_ptr.add(px) };
            let h = unsafe { src.read_unaligned() };
            let s = unsafe { src.add(1).read_unaligned() };
            let v = unsafe { src.add(2).read_unaligned() };

            let s_f = s as f32 * scale;
            let v_f = v as f32 * scale;

            let hx = x * channels;
            let rgb;
            match target {
                HsvTarget::HSV => {
                    let hsv = Hsv::from_components(h as f32, s_f, v_f);
                    rgb = hsv.to_rgb8();
                }
                HsvTarget::HSL => {
                    let hsl = Hsl::from_components(h as f32, s_f, v_f);
                    rgb = hsl.to_rgb8();
                }
            }

            unsafe {
                *dst_slice.get_unchecked_mut(hx + image_configuration.get_r_channel_offset()) =
                    rgb.r;
                *dst_slice.get_unchecked_mut(hx + image_configuration.get_g_channel_offset()) =
                    rgb.g;
                *dst_slice.get_unchecked_mut(hx + image_configuration.get_b_channel_offset()) =
                    rgb.b;
            }

            if image_configuration.has_alpha() {
                let a = unsafe { src.add(3).read_unaligned() };
                unsafe {
                    *dst_slice.get_unchecked_mut(hx + image_configuration.get_a_channel_offset()) =
                        a as u8;
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts HSV to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn hsv_to_rgb(
    src: &[u16],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    hsv_u16_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { HsvTarget::HSV as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts HSV to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn hsv_to_bgra(
    src: &[u16],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    hsv_u16_to_channels::<{ ImageConfiguration::Bgra as u8 }, true, { HsvTarget::HSV as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts HSV to RGBA. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn hsv_to_rgba(
    src: &[u16],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    hsv_u16_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { HsvTarget::HSV as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts HSL to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSL data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and L is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,L in range [0, scale]
pub fn hsl_to_rgb(
    src: &[u16],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    hsv_u16_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { HsvTarget::HSL as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts HSL to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSL data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn hsl_to_bgra(
    src: &[u16],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    hsv_u16_to_channels::<{ ImageConfiguration::Bgra as u8 }, true, { HsvTarget::HSL as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts HSL to RGBA. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSL data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn hsl_to_rgba(
    src: &[u16],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    hsv_u16_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { HsvTarget::HSL as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}
