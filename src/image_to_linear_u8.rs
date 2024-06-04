use std::slice;

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon_to_linear::get_neon_linear_transfer;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon_to_linear_u8::neon_image_linear_to_u8;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse_image_to_linear_u8::sse_image_to_linear_unsigned;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse_to_linear::get_sse_linear_transfer;
use crate::Rgb;

#[inline]
fn channels_to_linear<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Alpha may be set only on images with alpha");
        }
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let transfer = transfer_function.get_linearize_function();

    let channels = image_configuration.get_channels_count();

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let mut _has_sse = false;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if is_x86_feature_detected!("sse4.1") {
        _has_sse = true;
    }

    for _ in 0..height as usize {
        let mut cx = 0usize;

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            if _has_sse {
                let transfer = get_sse_linear_transfer(transfer_function);
                cx = sse_image_to_linear_unsigned::sse_channels_to_linear::<
                    CHANNELS_CONFIGURATION,
                    USE_ALPHA,
                >(
                    cx,
                    src.as_ptr(),
                    src_offset,
                    width,
                    dst.as_mut_ptr(),
                    dst_offset,
                    &transfer,
                )
            }
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            let transfer = get_neon_linear_transfer(transfer_function);
            cx = neon_image_linear_to_u8::neon_channels_to_linear_u8::<
                CHANNELS_CONFIGURATION,
                USE_ALPHA,
            >(
                cx,
                src.as_ptr(),
                src_offset,
                width,
                dst.as_mut_ptr(),
                dst_offset,
                &transfer,
            )
        }

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };
        let dst_slice: &mut [u8] =
            unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

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
            let rgb_f32 = rgb.to_rgb_f32();

            unsafe {
                let t_r = (transfer(rgb_f32.r) * 255f32).min(255f32).max(0f32) as u8;
                let t_g = (transfer(rgb_f32.g) * 255f32).min(255f32).max(0f32) as u8;
                let t_b = (transfer(rgb_f32.b) * 255f32).min(255f32).max(0f32) as u8;
                *dst_slice.get_unchecked_mut(px) = t_r;
                *dst_slice.get_unchecked_mut(px + 1) = t_g;
                *dst_slice.get_unchecked_mut(px + 2) = t_b;
            }

            if USE_ALPHA && image_configuration.has_alpha() {
                let a = unsafe {
                    *src_slice.get_unchecked(px + image_configuration.get_a_channel_offset())
                };
                unsafe {
                    *dst_slice.get_unchecked_mut(px + 3) = a;
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts RGB to Linear. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_linear_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear::<{ ImageConfiguration::Rgb as u8 }, false>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts RGBA to Linear, Alpha channel is normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgba_to_linear_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear::<{ ImageConfiguration::Rgba as u8 }, true>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts BGRA to Linear, Alpha channel is normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgra_to_linear_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear::<{ ImageConfiguration::Bgra as u8 }, true>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts BGR to Linear. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgr_to_linear_u8(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear::<{ ImageConfiguration::Bgr as u8 }, false>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
