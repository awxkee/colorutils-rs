use std::slice;

use crate::{Lab, Xyz, XYZ_TO_SRGB_D65};
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::image_to_xyz_lab::XyzTarget;
use crate::image_to_xyz_lab::XyzTarget::{LAB, XYZ};

fn xyz_to_channels<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool, const SOURCE: u8>(
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
    let source: XyzTarget = SOURCE.into();
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

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *mut f32 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * channels) };
        let dst_slice = unsafe { slice::from_raw_parts_mut(dst_ptr, width as usize * channels) };

        for x in cx..width as usize {
            let l_x = src_slice[x * 3];
            let l_y = src_slice[x * 3 + 1];
            let l_z = src_slice[x * 3 + 2];
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
            }

            dst_slice[x * channels + image_configuration.get_r_channel_offset()] = rgb.r;
            dst_slice[x * channels + image_configuration.get_g_channel_offset()] = rgb.g;
            dst_slice[x * channels + image_configuration.get_b_channel_offset()] = rgb.b;
            if image_configuration.has_alpha() {
                let a_ptr =
                    unsafe { (a_channel.as_ptr() as *const u8).add(a_offset) as *const f32 };
                let a_slice = unsafe { slice::from_raw_parts(a_ptr, width as usize) };
                let a_value = ((a_slice[x]) * 255f32).max(0f32);
                dst_slice[x * channels + image_configuration.get_a_channel_offset()] = a_value as u8;
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
