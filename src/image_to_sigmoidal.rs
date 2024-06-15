use std::slice;

use crate::image::ImageConfiguration;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_image_to_sigmoidal;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_image_to_sigmoidal_row;
use crate::Rgb;

#[inline]
fn image_to_sigmoidal<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
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

    const COLOR_SCALE: f32 = 1f32 / 255f32;

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        unsafe {
            cx = sse_image_to_sigmoidal_row::<CHANNELS_CONFIGURATION, USE_ALPHA>(
                cx, src_ptr, width, dst_ptr,
            );
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            cx = neon_image_to_sigmoidal::<CHANNELS_CONFIGURATION, USE_ALPHA>(
                cx,
                src.as_ptr(),
                src_offset,
                width,
                dst.as_mut_ptr(),
                dst_offset,
            )
        }

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

            let writing_ptr = unsafe { dst_ptr.add(hx) };

            let sigmoidal = rgb.to_sigmoidal();
            unsafe {
                writing_ptr.write_unaligned(sigmoidal.sr);
                writing_ptr.add(1).write_unaligned(sigmoidal.sg);
                writing_ptr.add(2).write_unaligned(sigmoidal.sb);
            }

            if image_configuration.has_alpha() {
                let a = unsafe {
                    *src_slice.get_unchecked(hx + image_configuration.get_a_channel_offset())
                } as f32
                    * COLOR_SCALE;

                unsafe {
                    writing_ptr.add(3).write_unaligned(a);
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts RGB to Sigmoidal. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
pub fn rgb_to_sigmoidal(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_sigmoidal::<{ ImageConfiguration::Rgb as u8 }, false>(
        src, src_stride, dst, dst_stride, width, height,
    );
}

/// This function converts BGRA to Sigmoidal. Alpha channel will be normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Sigmodal data
/// * `dst_stride` - Bytes per row for dst data
pub fn bgra_to_sigmoidal(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_sigmoidal::<{ ImageConfiguration::Bgra as u8 }, true>(
        src, src_stride, dst, dst_stride, width, height,
    );
}

/// This function converts RGBA to Sigmoidal. Alpha channel will be normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Sigmoidal data
/// * `dst_stride` - Bytes per row for dst data
pub fn rgba_to_sigmoidal(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_sigmoidal::<{ ImageConfiguration::Rgba as u8 }, true>(
        src, src_stride, dst, dst_stride, width, height,
    );
}
