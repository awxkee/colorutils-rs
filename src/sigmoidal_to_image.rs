use std::slice;

use crate::image::ImageConfiguration;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_from_sigmoidal_row;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_from_sigmoidal_row;
use crate::{Rgb, Sigmoidal};

#[inline]
fn sigmoidal_to_image<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Alpha may be set only on images with alpha");
        }
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

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

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *const f32 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        unsafe {
            _cx = sse_from_sigmoidal_row::<CHANNELS_CONFIGURATION>(_cx, src_ptr, dst_ptr, width);
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            _cx = neon_from_sigmoidal_row::<CHANNELS_CONFIGURATION>(_cx, src_ptr, dst_ptr, width);
        }

        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

        for x in _cx..width as usize {
            let px = x * channels;
            let reading_ptr = unsafe { src_ptr.add(px) };
            let sr = unsafe { reading_ptr.read_unaligned() };
            let sg = unsafe { reading_ptr.add(1).read_unaligned() };
            let sb = unsafe { reading_ptr.add(2).read_unaligned() };

            let sigmoidal = Sigmoidal::new(sr, sg, sb);
            let rgb: Rgb<u8> = sigmoidal.into();

            let hx = x * channels;

            unsafe {
                *dst_slice.get_unchecked_mut(hx + image_configuration.get_r_channel_offset()) =
                    rgb.r;
                *dst_slice.get_unchecked_mut(hx + image_configuration.get_g_channel_offset()) =
                    rgb.g;
                *dst_slice.get_unchecked_mut(hx + image_configuration.get_b_channel_offset()) =
                    rgb.b;
            }

            if image_configuration.has_alpha() {
                let a = (unsafe { reading_ptr.add(3).read_unaligned() } * 255f32).max(0f32);
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

/// This function converts Sigmoid to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    sigmoidal_to_image::<{ ImageConfiguration::Rgb as u8 }, false>(
        src, src_stride, dst, dst_stride, width, height,
    );
}

/// This function converts Sigmoid to BGRA. Alpha channel expected to be normalized and will be denormalized during transformation. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    sigmoidal_to_image::<{ ImageConfiguration::Bgra as u8 }, true>(
        src, src_stride, dst, dst_stride, width, height,
    );
}

/// This function converts Sigmoid to RGBA. Alpha channel expected to be normalized and will be denormalized during transformation. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    sigmoidal_to_image::<{ ImageConfiguration::Rgba as u8 }, true>(
        src, src_stride, dst, dst_stride, width, height,
    );
}
