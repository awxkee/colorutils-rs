use std::slice;

use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_channels_to_hsv_u16;
use crate::Rgb;

#[inline(always)]
fn channels_to_hsv_u16<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u16],
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

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let channels = image_configuration.get_channels_count();

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            cx = neon_channels_to_hsv_u16::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                cx,
                src.as_ptr(),
                src_offset,
                width,
                dst.as_mut_ptr(),
                dst_offset,
                scale,
            )
        }

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut u16 };

        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

        for x in cx..width as usize {
            let px = x * channels;
            let r = unsafe {
                *src_slice.get_unchecked(px + image_configuration.get_r_channel_offset())
            };
            let g = unsafe {
                *src_slice.get_unchecked(px + image_configuration.get_g_channel_offset())
            };
            let b = unsafe {
                *src_slice.get_unchecked(px + image_configuration.get_b_channel_offset())
            };

            let rgb = Rgb::<u8>::new(r, g, b);
            let hx = x * channels;
            match target {
                HsvTarget::HSV => {
                    let hsv = rgb.to_hsv();
                    unsafe {
                        *dst_slice.get_unchecked_mut(hx) = hsv.h as u16;
                        *dst_slice.get_unchecked_mut(hx + 1) = (hsv.s * scale).round() as u16;
                        *dst_slice.get_unchecked_mut(hx + 2) = (hsv.v * scale).round() as u16;
                    }
                }
                HsvTarget::HSL => {
                    let hsl = rgb.to_hsl();
                    unsafe {
                        *dst_slice.get_unchecked_mut(hx) = hsl.h as u16;
                        *dst_slice.get_unchecked_mut(hx + 1) = (hsl.s * scale).round() as u16;
                        *dst_slice.get_unchecked_mut(hx + 2) = (hsl.l * scale).round() as u16;
                    }
                }
            }

            if image_configuration.has_alpha() {
                let a = unsafe {
                    *src_slice.get_unchecked(hx + image_configuration.get_a_channel_offset())
                };
                unsafe {
                    *dst_slice.get_unchecked_mut(hx + 3) = a as u16;
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts RGB to HSV. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn rgb_to_hsv(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u16],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    channels_to_hsv_u16::<{ ImageConfiguration::Rgb as u8 }, false, { HsvTarget::HSV as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts BGRA to HSV. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn bgra_to_hsv(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u16],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    channels_to_hsv_u16::<{ ImageConfiguration::Bgra as u8 }, true, { HsvTarget::HSV as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts RGBA to HSV. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn rgba_to_hsv(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u16],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    channels_to_hsv_u16::<{ ImageConfiguration::Rgba as u8 }, true, { HsvTarget::HSV as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts RGB to HSL. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSL data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and L is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,L in range [0, scale]
pub fn rgb_to_hsl(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u16],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    channels_to_hsv_u16::<{ ImageConfiguration::Rgb as u8 }, false, { HsvTarget::HSL as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts BGRA to HSL. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSL data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn bgra_to_hsl(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u16],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    channels_to_hsv_u16::<{ ImageConfiguration::Bgra as u8 }, true, { HsvTarget::HSL as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}

/// This function converts RGBA to HSL. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSL data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn rgba_to_hsl(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u16],
    dst_stride: u32,
    width: u32,
    height: u32,
    scale: f32,
) {
    channels_to_hsv_u16::<{ ImageConfiguration::Rgba as u8 }, true, { HsvTarget::HSL as u8 }>(
        src, src_stride, dst, dst_stride, width, height, scale,
    );
}
