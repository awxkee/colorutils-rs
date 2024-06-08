use std::slice;

use crate::{Hsl, Hsv};
use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
use crate::neon::neon_hsv_u16_to_image;

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

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let channels = image_configuration.get_channels_count();

    let scale = 1f32 / scale;

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            cx = neon_hsv_u16_to_image::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                cx,
                src.as_ptr(),
                src_offset,
                width,
                dst.as_mut_ptr(),
                dst_offset,
                scale,
            )
        }

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *const u16 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

        for x in cx..width as usize {
            let px = x * channels;
            let h = unsafe { *src_slice.get_unchecked(px) };
            let s = unsafe { *src_slice.get_unchecked(px + 1) };
            let v = unsafe { *src_slice.get_unchecked(px + 2) };

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
                let a = unsafe { *src_slice.get_unchecked(hx + 3) };
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
