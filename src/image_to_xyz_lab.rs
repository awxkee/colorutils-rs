#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(unused_imports)]
use crate::avx2_to_xyz_lab::*;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::image_to_xyz_lab::XyzTarget::{LAB, XYZ};
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon_to_xyz_lab::neon_channels_to_xyz_or_lab;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse_to_xyz_lab::sse_channels_to_xyz_or_lab;
use crate::{Rgb, Xyz, SRGB_TO_XYZ_D65};
use std::slice;

pub(crate) enum XyzTarget {
    LAB = 0,
    XYZ = 1,
}

impl From<u8> for XyzTarget {
    fn from(value: u8) -> Self {
        match value {
            0 => LAB,
            1 => XYZ,
            _ => {
                panic!("Not implemented")
            }
        }
    }
}

#[inline(always)]
fn channels_to_xyz<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool, const TARGET: u8>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_channel: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let target: XyzTarget = TARGET.into();
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

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let mut _has_sse = false;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let mut _has_avx2 = false;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        #[cfg(target_feature = "avx2")]
        if is_x86_feature_detected!("avx2") {
            _has_avx2 = true;
        }
        if is_x86_feature_detected!("sse4.1") {
            _has_sse = true;
        }
    }

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            if _has_avx2 {
                if USE_ALPHA {
                    cx = avx2_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                        cx,
                        src.as_ptr(),
                        src_offset,
                        width,
                        dst.as_mut_ptr(),
                        dst_offset,
                        a_channel.as_mut_ptr(),
                        a_offset,
                        &matrix,
                        transfer_function,
                    );
                } else {
                    cx = avx2_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                        cx,
                        src.as_ptr(),
                        src_offset,
                        width,
                        dst.as_mut_ptr(),
                        dst_offset,
                        std::ptr::null_mut(),
                        0usize,
                        &matrix,
                        transfer_function,
                    );
                }
            } else if _has_sse {
                if USE_ALPHA {
                    cx = sse_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                        cx,
                        src.as_ptr(),
                        src_offset,
                        width,
                        dst.as_mut_ptr(),
                        dst_offset,
                        a_channel.as_mut_ptr(),
                        a_offset,
                        &matrix,
                        transfer_function,
                    )
                } else {
                    cx = sse_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                        cx,
                        src.as_ptr(),
                        src_offset,
                        width,
                        dst.as_mut_ptr(),
                        dst_offset,
                        std::ptr::null_mut(),
                        0usize,
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
                cx = neon_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                    cx,
                    src.as_ptr(),
                    src_offset,
                    width,
                    dst.as_mut_ptr(),
                    dst_offset,
                    a_channel.as_mut_ptr(),
                    a_offset,
                    &matrix,
                    transfer_function,
                )
            } else {
                cx = neon_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>(
                    cx,
                    src.as_ptr(),
                    src_offset,
                    width,
                    dst.as_mut_ptr(),
                    dst_offset,
                    std::ptr::null_mut(),
                    0usize,
                    &matrix,
                    transfer_function,
                )
            }
        }

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * 3) };

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
            match target {
                LAB => {
                    let lab = rgb.to_lab();
                    unsafe {
                        *dst_slice.get_unchecked_mut(x * 3) = lab.l;
                        *dst_slice.get_unchecked_mut(x * 3 + 1) = lab.a;
                        *dst_slice.get_unchecked_mut(x * 3 + 2) = lab.b;
                    }
                }
                XYZ => {
                    let xyz = Xyz::from_rgb(&rgb, &matrix, transfer_function);
                    unsafe {
                        *dst_slice.get_unchecked_mut(x * 3) = xyz.x;
                        *dst_slice.get_unchecked_mut(x * 3 + 1) = xyz.y;
                        *dst_slice.get_unchecked_mut(x * 3 + 2) = xyz.z;
                    }
                }
            }

            if USE_ALPHA && image_configuration.has_alpha() {
                let a = unsafe {
                    *src_slice.get_unchecked(px + image_configuration.get_a_channel_offset())
                };
                let a_lin = a as f32 * (1f32 / 255f32);
                let a_ptr =
                    unsafe { (a_channel.as_mut_ptr() as *mut u8).add(a_offset) as *mut f32 };
                let a_slice = unsafe { slice::from_raw_parts_mut(a_ptr, width as usize) };
                unsafe {
                    *a_slice.get_unchecked_mut(x) = a_lin;
                }
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
        a_offset += a_stride as usize;
    }
}

/// This function converts RGB to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_xyz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XYZ as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts sRGB D65 to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn srgb_to_xyz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XYZ as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts RGB to CIE L*ab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
pub fn rgb_to_lab(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts RGBA to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgba_to_xyz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XYZ as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts sRGB RGBA D65 to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGBA to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn srgba_to_xyz(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XYZ as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts RGBA to XYZ with preserving and linearizing alpha channels. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgba_to_xyza(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { XYZ as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts RGBA to XYZ with preserving and linearizing alpha channels. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
pub fn srgba_to_xyza(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { XYZ as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts RGBA to CIE L*ab against D65 white point without alpha. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
pub fn rgba_to_lab(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts RGBA to CIE L*ab against D65 white point and preserving and normalizing alpha channels. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
pub fn rgba_to_laba(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts BGRA to CIE L*ab against D65 white point and preserving and linearizing alpha channels. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
pub fn bgra_to_laba(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz::<{ ImageConfiguration::Bgra as u8 }, true, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts BGR to CIE L*ab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
pub fn bgr_to_lab(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { LAB as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
}
