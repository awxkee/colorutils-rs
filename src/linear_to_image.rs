#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
use crate::avx::avx_linear_to_gamma;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_linear_to_gamma;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_linear_to_gamma;
use crate::Rgb;

#[inline(always)]
fn linear_to_gamma_channels<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &[f32],
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

    let transfer = transfer_function.get_gamma_function();

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

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx2"
    ))]
    let mut _has_avx2 = false;

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx2"
    ))]
    if is_x86_feature_detected!("avx2") {
        _has_avx2 = true;
    }

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2"
        ))]
        unsafe {
            if _has_avx2 {
                _cx = avx_linear_to_gamma::<CHANNELS_CONFIGURATION, USE_ALPHA>(
                    _cx,
                    src.as_ptr(),
                    src_offset as u32,
                    dst.as_mut_ptr(),
                    dst_offset as u32,
                    width,
                    transfer_function,
                )
            }
        }

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        unsafe {
            if _has_sse {
                _cx = sse_linear_to_gamma::<CHANNELS_CONFIGURATION, USE_ALPHA>(
                    _cx,
                    src.as_ptr(),
                    src_offset as u32,
                    dst.as_mut_ptr(),
                    dst_offset as u32,
                    width,
                    transfer_function,
                )
            }
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            _cx = neon_linear_to_gamma::<CHANNELS_CONFIGURATION, USE_ALPHA>(
                _cx,
                src.as_ptr(),
                src_offset as u32,
                dst.as_mut_ptr(),
                dst_offset as u32,
                width,
                transfer_function,
            );
        }

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *const f32 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        for x in _cx..width as usize {
            let px = x * channels;
            let src_slice = unsafe { src_ptr.add(px) };
            let r = unsafe {
                src_slice
                    .add(image_configuration.get_r_channel_offset())
                    .read_unaligned()
            };
            let g = unsafe {
                src_slice
                    .add(image_configuration.get_g_channel_offset())
                    .read_unaligned()
            };
            let b = unsafe {
                src_slice
                    .add(image_configuration.get_b_channel_offset())
                    .read_unaligned()
            };

            let rgb = Rgb::<f32>::new(
                r.min(1f32).max(0f32),
                g.min(1f32).max(0f32),
                b.min(1f32).max(0f32),
            );

            let dst = unsafe { dst_ptr.add(px) };

            unsafe {
                dst.write_unaligned((transfer(rgb.r).round() * 255f32) as u8);
                dst.add(1)
                    .write_unaligned((transfer(rgb.g).round() * 255f32) as u8);
                dst.add(2)
                    .write_unaligned((transfer(rgb.b).round() * 255f32) as u8);
            }

            if USE_ALPHA && image_configuration.has_alpha() {
                let a = unsafe {
                    src_slice
                        .add(image_configuration.get_a_channel_offset())
                        .read_unaligned()
                };
                let a_lin = (a * 255f32).round() as u8;
                unsafe {
                    dst.add(3).write_unaligned(a_lin);
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts Linear to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive Gamma RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels::<{ ImageConfiguration::Rgb as u8 }, false>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Linear RGBA to RGBA, Alpha channel will be denormalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels::<{ ImageConfiguration::Rgba as u8 }, true>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Linear BGRA to BGRA, Alpha channel will de dernormalizaed. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive Gamma BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels::<{ ImageConfiguration::Bgra as u8 }, true>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

/// This function converts Linear BGR to Gamma BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels::<{ ImageConfiguration::Bgr as u8 }, false>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}
