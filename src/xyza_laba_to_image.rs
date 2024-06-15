use std::slice;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
use crate::avx::avx_xyza_to_image;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::image_to_xyz_lab::XyzTarget;
use crate::image_to_xyz_lab::XyzTarget::{LAB, LUV, XYZ};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_xyza_to_image;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_xyza_to_image;
use crate::{Lab, Luv, Xyz, XYZ_TO_SRGB_D65};

fn xyz_with_alpha_to_channels<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let source: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    let mut _has_sse = false;

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

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    if is_x86_feature_detected!("sse4.1") {
        _has_sse = true;
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let channels = image_configuration.get_channels_count();

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2"
        ))]
        unsafe {
            if _has_avx2 {
                cx = avx_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>(
                    cx,
                    src.as_ptr(),
                    src_offset,
                    dst.as_mut_ptr(),
                    dst_offset,
                    width,
                    &matrix,
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
                cx = sse_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>(
                    cx,
                    src.as_ptr(),
                    src_offset,
                    dst.as_mut_ptr(),
                    dst_offset,
                    width,
                    &matrix,
                    transfer_function,
                )
            }
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            cx = neon_xyza_to_image::<CHANNELS_CONFIGURATION, TARGET>(
                cx,
                src.as_ptr(),
                src_offset,
                dst.as_mut_ptr(),
                dst_offset,
                width,
                &matrix,
                transfer_function,
            )
        }

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *mut f32 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

        for x in cx..width as usize {
            let px = x * 4;
            let l_x = unsafe { *src_slice.get_unchecked(px) };
            let l_y = unsafe { *src_slice.get_unchecked(px + 1) };
            let l_z = unsafe { *src_slice.get_unchecked(px + 2) };
            let rgb;
            match source {
                LAB => {
                    let lab = Lab::new(l_x, l_y, l_z);
                    rgb = lab.to_rgb();
                }
                XYZ => {
                    let xyz = Xyz::new(l_x, l_y, l_z);
                    rgb = xyz.to_rgb(&matrix, transfer_function);
                }
                LUV => {
                    let luv = Luv::new(l_x, l_y, l_z);
                    rgb = luv.to_rgb();
                }
            }

            let l_a = unsafe { *src_slice.get_unchecked(px + 3) };
            let a_value = (l_a * 255f32).max(0f32);
            unsafe {
                *dst_slice
                    .get_unchecked_mut(x * channels + image_configuration.get_r_channel_offset()) =
                    rgb.r;
                *dst_slice
                    .get_unchecked_mut(x * channels + image_configuration.get_g_channel_offset()) =
                    rgb.g;
                *dst_slice
                    .get_unchecked_mut(x * channels + image_configuration.get_b_channel_offset()) =
                    rgb.b;
                dst_slice[x * channels + image_configuration.get_a_channel_offset()] =
                    a_value as u8;
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts LAB with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
pub fn lab_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts LAB with separate alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
pub fn lab_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts LUV with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
pub fn luv_with_alpha_to_rgba(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Rgba as u8 }, { LUV as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts LUV with separate alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
pub fn luv_with_alpha_to_bgra(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_with_alpha_to_channels::<{ ImageConfiguration::Bgra as u8 }, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}
