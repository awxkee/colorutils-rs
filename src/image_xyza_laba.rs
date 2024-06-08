use crate::image::ImageConfiguration;
use crate::image_to_xyz_lab::XyzTarget;
use crate::image_to_xyz_lab::XyzTarget::{LAB, LUV, XYZ};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_channels_to_xyza_or_laba;
use crate::{Rgb, TransferFunction, Xyz, SRGB_TO_XYZ_D65};
use std::slice;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_channels_to_xyza_laba;

#[inline(always)]
fn channels_to_xyz_with_alpha<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let mut _has_sse = false;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        #[cfg(target_feature = "sse4.1")]
        if is_x86_feature_detected!("sse4.1") {
            _has_sse = true;
        }
    }

    const CHANNELS: usize = 4;

    let channels = image_configuration.get_channels_count();

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            if _has_sse {
                cx = sse_channels_to_xyza_laba::<CHANNELS_CONFIGURATION, TARGET>(
                    cx,
                    src.as_ptr(),
                    src_offset,
                    width,
                    dst.as_mut_ptr(),
                    dst_offset,
                    &matrix,
                    transfer_function,
                );
            }
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            cx = neon_channels_to_xyza_or_laba::<CHANNELS_CONFIGURATION, TARGET>(
                cx,
                src.as_ptr(),
                src_offset,
                width,
                dst.as_mut_ptr(),
                dst_offset,
                &matrix,
                transfer_function,
            )
        }

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * 4) };

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
            let px = x * CHANNELS;
            match target {
                LAB => {
                    let lab = rgb.to_lab();
                    unsafe {
                        *dst_slice.get_unchecked_mut(px) = lab.l;
                        *dst_slice.get_unchecked_mut(px + 1) = lab.a;
                        *dst_slice.get_unchecked_mut(px + 2) = lab.b;
                    }
                }
                XYZ => {
                    let xyz = Xyz::from_rgb(&rgb, &matrix, transfer_function);
                    unsafe {
                        *dst_slice.get_unchecked_mut(px) = xyz.x;
                        *dst_slice.get_unchecked_mut(px + 1) = xyz.y;
                        *dst_slice.get_unchecked_mut(px + 2) = xyz.z;
                    }
                }
                XyzTarget::LUV => {
                    let luv = rgb.to_luv();
                    unsafe {
                        *dst_slice.get_unchecked_mut(px) = luv.l;
                        *dst_slice.get_unchecked_mut(px + 1) = luv.u;
                        *dst_slice.get_unchecked_mut(px + 2) = luv.v;
                    }
                }
            }

            let a = unsafe {
                *src_slice.get_unchecked(px + image_configuration.get_a_channel_offset())
            };
            let a_lin = a as f32 * (1f32 / 255f32);
            unsafe {
                *dst_slice.get_unchecked_mut(x * CHANNELS + 3) = a_lin;
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts RGBA to CIE L*ab against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
pub fn rgba_to_lab_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts BGRA to CIE L*ab against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
pub fn bgra_to_lab_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts RGBA to CIE L*uv against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
pub fn rgba_to_luv_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { LUV as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts BGRA to CIE L*uv against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
pub fn bgra_to_luv_with_alpha(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { LUV as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}
