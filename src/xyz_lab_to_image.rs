use std::slice;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
use crate::avx::avx_xyz_to_channels;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::image_to_xyz_lab::XyzTarget;
use crate::image_to_xyz_lab::XyzTarget::{LAB, LUV, XYZ};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::neon_xyz_to_channels;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_xyz_to_channels;
use crate::{Lab, Luv, Xyz, XYZ_TO_SRGB_D65};

fn xyz_to_channels<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool, const TARGET: u8>(
    src: &[f32],
    src_stride: u32,
    a_channel: &[f32],
    a_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let source: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA {
        if !image_configuration.has_alpha() {
            panic!("Alpha may be set only on images with alpha");
        }
    }

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;
    let mut a_offset = 0usize;

    let channels = image_configuration.get_channels_count();

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

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2"
        ))]
        unsafe {
            if _has_avx2 {
                if USE_ALPHA {
                    cx = avx_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                        cx,
                        src.as_ptr(),
                        src_offset,
                        a_channel.as_ptr(),
                        a_offset,
                        dst.as_mut_ptr(),
                        dst_offset,
                        width,
                        &matrix,
                        transfer_function,
                    )
                } else {
                    cx = avx_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                        cx,
                        src.as_ptr(),
                        src_offset,
                        std::ptr::null(),
                        0usize,
                        dst.as_mut_ptr(),
                        dst_offset,
                        width,
                        &matrix,
                        transfer_function,
                    )
                }
            }
        }

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        unsafe {
            if _has_sse {
                if USE_ALPHA {
                    cx = sse_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                        cx,
                        src.as_ptr(),
                        src_offset,
                        a_channel.as_ptr(),
                        a_offset,
                        dst.as_mut_ptr(),
                        dst_offset,
                        width,
                        &matrix,
                        transfer_function,
                    )
                } else {
                    cx = sse_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                        cx,
                        src.as_ptr(),
                        src_offset,
                        std::ptr::null(),
                        0usize,
                        dst.as_mut_ptr(),
                        dst_offset,
                        width,
                        &matrix,
                        transfer_function,
                    )
                }
            }
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            if USE_ALPHA {
                cx = neon_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                    cx,
                    src.as_ptr(),
                    src_offset,
                    a_channel.as_ptr(),
                    a_offset,
                    dst.as_mut_ptr(),
                    dst_offset,
                    width,
                    &matrix,
                    transfer_function,
                )
            } else {
                cx = neon_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                    cx,
                    src.as_ptr(),
                    src_offset,
                    std::ptr::null(),
                    0usize,
                    dst.as_mut_ptr(),
                    dst_offset,
                    width,
                    &matrix,
                    transfer_function,
                )
            }
        }

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *mut f32 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

        for x in cx..width as usize {
            let src_slice = unsafe { src_ptr.add(x * 3) };
            let l_x = unsafe { src_slice.read_unaligned() };
            let l_y = unsafe { src_slice.add(1).read_unaligned() };
            let l_z = unsafe { src_slice.add(2).read_unaligned() };
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
                XyzTarget::LUV => {
                    let luv = Luv::new(l_x, l_y, l_z);
                    rgb = luv.to_rgb();
                }
            }

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
            }
            if image_configuration.has_alpha() {
                let a_ptr =
                    unsafe { (a_channel.as_ptr() as *const u8).add(a_offset) as *const f32 };
                let a_f = unsafe { a_ptr.add(x).read_unaligned() };
                let a_value = (a_f * 255f32).max(0f32);
                unsafe {
                    *dst_slice.get_unchecked_mut(
                        x * channels + image_configuration.get_a_channel_offset(),
                    ) = a_value as u8;
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
        a_offset += a_stride as usize;
    }
}

/// This function converts XYZ to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains XYZ data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn xyz_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XYZ as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts XYZ to sRGB D65 White point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains XYZ data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn xyz_to_srgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XYZ as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts LAB to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn lab_to_srgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { LAB as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts LAB with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
pub fn laba_to_srgb(
    src: &[f32],
    src_stride: u32,
    a_plane: &[f32],
    a_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { LAB as u8 }>(
        src,
        src_stride,
        &a_plane,
        a_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts XYZ with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
pub fn xyza_to_rgba(
    src: &[f32],
    src_stride: u32,
    a_plane: &[f32],
    a_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { XYZ as u8 }>(
        src,
        src_stride,
        &a_plane,
        a_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts LUV to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn luv_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { LUV as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts LUV to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn luv_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { LUV as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}
