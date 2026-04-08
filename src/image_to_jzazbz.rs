/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_image_to_jzazbz;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_image_to_jzazbz;
use crate::{
    ColorError, ImageBuffer, ImageBufferMut, Jzazbz, Jzczhz, Rgb, TransferFunction, Xyz,
    SRGB_TO_XYZ_D65,
};

#[repr(u8)]
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum JzazbzTarget {
    Jzazbz = 0,
    Jzczhz = 1,
}

impl From<u8> for JzazbzTarget {
    fn from(value: u8) -> Self {
        match value {
            0 => JzazbzTarget::Jzazbz,
            1 => JzazbzTarget::Jzczhz,
            _ => {
                panic!("Not known value {}", value)
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn channels_to_jzaz<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let target: JzazbzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let mut _wide_row_handle: Option<unsafe fn(usize, &[f32], &mut [f32], u32, f32) -> usize> =
        None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handle = Some(neon_image_to_jzazbz::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_image_to_jzazbz::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let mut lut_table = [0f32; 256];
    for (i, table) in lut_table.iter_mut().enumerate() {
        *table = transfer_function.linearize(i as f32 * (1. / 255.0));
    }

    let mut a_lut_table = [0f32; 256];
    for (i, lut) in a_lut_table.iter_mut().enumerate() {
        *lut = i as f32 * (1. / 255.0);
    }

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;
    let width = src.width;

    let mut linearized_row = vec![0f32; src_r_width as usize];

    for (dst, src) in dst
        .data
        .borrow_mut()
        .chunks_mut(dst_stride)
        .zip(src.data.chunks(src.stride()))
    {
        let src = &src[..src_r_width as usize];
        let dst = &mut dst[..dst_r_width as usize];

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            for (dst, src) in linearized_row
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                dst[0] = lut_table[src[image_configuration.r_index()] as usize];
                dst[1] = lut_table[src[image_configuration.g_index()] as usize];
                dst[2] = lut_table[src[image_configuration.b_index()] as usize];
            }
        } else if image_configuration == ImageConfiguration::Bgra
            || image_configuration == ImageConfiguration::Rgba
        {
            for (dst, src) in linearized_row
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                dst[0] = lut_table[src[image_configuration.r_index()] as usize];
                dst[1] = lut_table[src[image_configuration.g_index()] as usize];
                dst[2] = lut_table[src[image_configuration.b_index()] as usize];
                dst[3] = a_lut_table[src[image_configuration.a_index()] as usize];
            }
        }

        let mut cx = 0usize;

        if let Some(dispatcher) = _wide_row_handle {
            cx = unsafe { dispatcher(cx, &linearized_row, dst, width, display_luminance) };
        }

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            let linearized_row = &linearized_row[cx * 3..];
            let dst = &mut dst[cx * 3..];
            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(linearized_row.as_chunks::<3>().0.iter())
            {
                let xyz =
                    Xyz::from_linear_rgb(Rgb::<f32>::new(src[0], src[1], src[2]), &SRGB_TO_XYZ_D65);
                match target {
                    JzazbzTarget::Jzazbz => {
                        let jzazbz =
                            Jzazbz::from_xyz_with_display_luminance(xyz, display_luminance);

                        dst[0] = jzazbz.jz;
                        dst[1] = jzazbz.az;
                        dst[2] = jzazbz.bz;
                    }
                    JzazbzTarget::Jzczhz => {
                        let jzczhz =
                            Jzczhz::from_xyz_with_display_luminance(xyz, display_luminance);

                        dst[0] = jzczhz.jz;
                        dst[1] = jzczhz.cz;
                        dst[2] = jzczhz.hz;
                    }
                }
            }
        } else if image_configuration == ImageConfiguration::Bgra
            || image_configuration == ImageConfiguration::Rgba
        {
            let linearized_row = &linearized_row[cx * 4..];
            let dst = &mut dst[cx * 4..];
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(linearized_row.as_chunks::<4>().0.iter())
            {
                let xyz =
                    Xyz::from_linear_rgb(Rgb::<f32>::new(src[0], src[1], src[2]), &SRGB_TO_XYZ_D65);
                match target {
                    JzazbzTarget::Jzazbz => {
                        let jzazbz =
                            Jzazbz::from_xyz_with_display_luminance(xyz, display_luminance);

                        dst[0] = jzazbz.jz;
                        dst[1] = jzazbz.az;
                        dst[2] = jzazbz.bz;
                    }
                    JzazbzTarget::Jzczhz => {
                        let jzczhz =
                            Jzczhz::from_xyz_with_display_luminance(xyz, display_luminance);

                        dst[0] = jzczhz.jz;
                        dst[1] = jzczhz.cz;
                        dst[2] = jzczhz.hz;
                    }
                }
                dst[3] = src[3];
            }
        }
    }
    Ok(())
}

/// This function converts RGB to Jzazbz against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzazbz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_jzazbz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_jzaz::<{ ImageConfiguration::Rgb as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts RGBA to Jzazbz against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzazbz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_jzazbz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_jzaz::<{ ImageConfiguration::Rgba as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts BGRA to Jzazbz against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzazbz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_jzazbz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_jzaz::<{ ImageConfiguration::Bgra as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts BGR to Jzazbz against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzazbz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_jzazbz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_jzaz::<{ ImageConfiguration::Bgr as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts RGB to Jzczhz against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzczhz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_jzczhz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_jzaz::<{ ImageConfiguration::Rgb as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts RGBA to Jzczhz against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzczhz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_jzczhz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_jzaz::<{ ImageConfiguration::Rgba as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts BGRA to Jzczhz against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzczhz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_jzczhz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_jzaz::<{ ImageConfiguration::Bgra as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts BGR to Jzczhz against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Jzczhz(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `display_luminance` - Target display luminance
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_jzczhz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_jzaz::<{ ImageConfiguration::Bgr as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BufferStore;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_src_u8(data: Vec<u8>, width: u32, channels: u32) -> ImageBuffer<'static, u8> {
        ImageBuffer {
            data: std::borrow::Cow::Owned(data),
            width,
            height: 1,
            channels,
            stride: width * channels,
        }
    }

    fn make_dst_f32(width: u32, channels: u32) -> ImageBufferMut<'static, f32> {
        ImageBufferMut {
            data: BufferStore::Owned(vec![0f32; (width * channels) as usize]),
            width,
            height: 1,
            channels,
            stride: width * channels,
        }
    }

    const DISPLAY_LUMINANCE: f32 = 200.0;

    #[test]
    fn test_rgb_to_jzazbz_black() {
        // Black (0,0,0) => Jz should be ~0, az/bz ~0
        let src = make_src_u8(vec![0, 0, 0], 1, 3);
        let mut dst = make_dst_f32(1, 3);
        rgb_to_jzazbz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        assert!(
            dst.data[0].abs() < 0.01,
            "Jz ~0 for black, got {}",
            dst.data[0]
        );
        assert!(
            dst.data[1].abs() < 0.01,
            "az ~0 for black, got {}",
            dst.data[1]
        );
        assert!(
            dst.data[2].abs() < 0.01,
            "bz ~0 for black, got {}",
            dst.data[2]
        );
    }

    #[test]
    fn test_rgb_to_jzazbz_white() {
        // White (255,255,255) => Jz should be > 0, az/bz ~0 (neutral)
        let src = make_src_u8(vec![255, 255, 255], 1, 3);
        let mut dst = make_dst_f32(1, 3);
        rgb_to_jzazbz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        assert!(
            dst.data[0] > 0.0,
            "Jz should be > 0 for white, got {}",
            dst.data[0]
        );
        assert!(
            dst.data[1].abs() < 0.01,
            "az ~0 for white, got {}",
            dst.data[1]
        );
        assert!(
            dst.data[2].abs() < 0.01,
            "bz ~0 for white, got {}",
            dst.data[2]
        );
    }

    #[test]
    fn test_rgb_to_jzazbz_white_brighter_than_black() {
        let src_black = make_src_u8(vec![0, 0, 0], 1, 3);
        let src_white = make_src_u8(vec![255, 255, 255], 1, 3);
        let mut dst_black = make_dst_f32(1, 3);
        let mut dst_white = make_dst_f32(1, 3);
        rgb_to_jzazbz(
            &src_black,
            &mut dst_black,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        rgb_to_jzazbz(
            &src_white,
            &mut dst_white,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert!(
            dst_white.data[0] > dst_black.data[0],
            "White Jz should exceed black Jz"
        );
    }

    #[test]
    fn test_rgb_to_jzazbz_multiple_pixels() {
        let src = make_src_u8(vec![0, 0, 0, 255, 255, 255], 2, 3);
        let mut dst = make_dst_f32(2, 3);
        rgb_to_jzazbz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        // pixel 0 = black
        assert!(dst.data[0].abs() < 0.01);
        // pixel 1 = white
        assert!(dst.data[3] > 0.0);
    }

    #[test]
    fn test_rgb_to_jzazbz_red_has_positive_az() {
        // Red in Jzazbz: az should be positive (red-green axis)
        let src = make_src_u8(vec![255, 0, 0], 1, 3);
        let mut dst = make_dst_f32(1, 3);
        rgb_to_jzazbz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        assert!(
            dst.data[1] > 0.0,
            "az should be positive for red, got {}",
            dst.data[1]
        );
    }

    #[test]
    fn test_rgb_to_jzazbz_display_luminance_affects_jz() {
        let src = make_src_u8(vec![128, 128, 128], 1, 3);
        let mut dst_low = make_dst_f32(1, 3);
        let mut dst_high = make_dst_f32(1, 3);
        rgb_to_jzazbz(&src, &mut dst_low, 100.0, TransferFunction::Srgb).unwrap();
        rgb_to_jzazbz(&src, &mut dst_high, 1000.0, TransferFunction::Srgb).unwrap();
        assert_ne!(
            dst_low.data[0], dst_high.data[0],
            "Different display luminance should yield different Jz"
        );
    }

    // ── rgba_to_jzazbz ───────────────────────────────────────────────────────

    #[test]
    fn test_rgba_to_jzazbz_alpha_normalized() {
        // Alpha 255 => normalized to 1.0
        let src = make_src_u8(vec![255, 255, 255, 255], 1, 4);
        let mut dst = make_dst_f32(1, 4);
        rgba_to_jzazbz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        let alpha = dst.data[3];
        assert!(
            (alpha - 1.0).abs() < 0.01,
            "Alpha should be ~1.0, got {alpha}"
        );
    }

    #[test]
    fn test_rgba_to_jzazbz_zero_alpha() {
        let src = make_src_u8(vec![255, 255, 255, 0], 1, 4);
        let mut dst = make_dst_f32(1, 4);
        rgba_to_jzazbz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        assert!(
            dst.data[3].abs() < 0.01,
            "Alpha should be ~0.0, got {}",
            dst.data[3]
        );
    }

    #[test]
    fn test_rgba_to_jzazbz_half_alpha() {
        let src = make_src_u8(vec![0, 0, 0, 128], 1, 4);
        let mut dst = make_dst_f32(1, 4);
        rgba_to_jzazbz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        let alpha = dst.data[3];
        assert!(
            (alpha - 128.0 / 255.0).abs() < 0.01,
            "Alpha ~0.502, got {alpha}"
        );
    }

    // ── bgr_to_jzazbz ────────────────────────────────────────────────────────

    #[test]
    fn test_bgr_to_jzazbz_matches_rgb_for_neutral() {
        // For neutral grey RGB==BGR so results must match
        let src_rgb = make_src_u8(vec![128, 128, 128], 1, 3);
        let src_bgr = make_src_u8(vec![128, 128, 128], 1, 3);
        let mut dst_rgb = make_dst_f32(1, 3);
        let mut dst_bgr = make_dst_f32(1, 3);
        rgb_to_jzazbz(
            &src_rgb,
            &mut dst_rgb,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgr_to_jzazbz(
            &src_bgr,
            &mut dst_bgr,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert!((dst_rgb.data[0] - dst_bgr.data[0]).abs() < 1e-5);
        assert!((dst_rgb.data[1] - dst_bgr.data[1]).abs() < 1e-5);
        assert!((dst_rgb.data[2] - dst_bgr.data[2]).abs() < 1e-5);
    }

    #[test]
    fn test_bgr_to_jzazbz_red_channel_swap() {
        // Pure red in RGB = (255,0,0); pure red in BGR = (0,0,255)
        // Both should give same Jz/az/bz since they represent the same color
        let src_rgb = make_src_u8(vec![255, 0, 0], 1, 3);
        let src_bgr = make_src_u8(vec![0, 0, 255], 1, 3);
        let mut dst_rgb = make_dst_f32(1, 3);
        let mut dst_bgr = make_dst_f32(1, 3);
        rgb_to_jzazbz(
            &src_rgb,
            &mut dst_rgb,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgr_to_jzazbz(
            &src_bgr,
            &mut dst_bgr,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert!(
            (dst_rgb.data[0] - dst_bgr.data[0]).abs() < 1e-4,
            "Jz mismatch"
        );
        assert!(
            (dst_rgb.data[1] - dst_bgr.data[1]).abs() < 1e-4,
            "az mismatch"
        );
        assert!(
            (dst_rgb.data[2] - dst_bgr.data[2]).abs() < 1e-4,
            "bz mismatch"
        );
    }

    // ── bgra_to_jzazbz ───────────────────────────────────────────────────────

    #[test]
    fn test_bgra_to_jzazbz_alpha_preserved() {
        let src = make_src_u8(vec![0, 0, 255, 200], 1, 4); // BGR red + alpha 200
        let mut dst = make_dst_f32(1, 4);
        bgra_to_jzazbz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        let alpha = dst.data[3];
        assert!(
            (alpha - 200.0 / 255.0).abs() < 0.01,
            "Alpha ~0.784, got {alpha}"
        );
    }

    // ── rgb_to_jzczhz ────────────────────────────────────────────────────────

    #[test]
    fn test_rgb_to_jzczhz_black() {
        let src = make_src_u8(vec![0, 0, 0], 1, 3);
        let mut dst = make_dst_f32(1, 3);
        rgb_to_jzczhz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        assert!(dst.data[0].abs() < 0.01, "Jz ~0 for black");
        assert!(dst.data[1].abs() < 0.01, "Cz ~0 for black");
    }

    #[test]
    fn test_rgb_to_jzczhz_white_low_chroma() {
        // White is neutral — chroma (Cz) should be near 0
        let src = make_src_u8(vec![255, 255, 255], 1, 3);
        let mut dst = make_dst_f32(1, 3);
        rgb_to_jzczhz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        assert!(dst.data[0] > 0.0, "Jz > 0 for white");
        assert!(
            dst.data[1].abs() < 0.01,
            "Cz ~0 for white, got {}",
            dst.data[1]
        );
    }

    #[test]
    fn test_rgb_to_jzczhz_saturated_has_chroma() {
        // A saturated colour should have Cz > 0
        let src = make_src_u8(vec![255, 0, 0], 1, 3);
        let mut dst = make_dst_f32(1, 3);
        rgb_to_jzczhz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        assert!(
            dst.data[1] > 0.0,
            "Cz should be > 0 for red, got {}",
            dst.data[1]
        );
    }

    #[test]
    fn test_rgb_to_jzczhz_hue_in_range() {
        // Hz (hue) must be in [0, 2π)
        let src = make_src_u8(vec![255, 0, 0], 1, 3);
        let mut dst = make_dst_f32(1, 3);
        rgb_to_jzczhz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        let hz = dst.data[2];
        assert!(
            hz >= 0.0 && hz < std::f32::consts::TAU,
            "Hz out of range: {hz}"
        );
    }

    // ── rgba_to_jzczhz ───────────────────────────────────────────────────────

    #[test]
    fn test_rgba_to_jzczhz_alpha_passthrough() {
        let src = make_src_u8(vec![255, 0, 0, 180], 1, 4);
        let mut dst = make_dst_f32(1, 4);
        rgba_to_jzczhz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        let alpha = dst.data[3];
        assert!(
            (alpha - 180.0 / 255.0).abs() < 0.01,
            "Alpha ~0.706, got {alpha}"
        );
    }

    // ── bgr_to_jzczhz ────────────────────────────────────────────────────────

    #[test]
    fn test_bgr_to_jzczhz_neutral_matches_rgb() {
        let src_rgb = make_src_u8(vec![100, 100, 100], 1, 3);
        let src_bgr = make_src_u8(vec![100, 100, 100], 1, 3);
        let mut dst_rgb = make_dst_f32(1, 3);
        let mut dst_bgr = make_dst_f32(1, 3);
        rgb_to_jzczhz(
            &src_rgb,
            &mut dst_rgb,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgr_to_jzczhz(
            &src_bgr,
            &mut dst_bgr,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert!((dst_rgb.data[0] - dst_bgr.data[0]).abs() < 1e-5);
        assert!((dst_rgb.data[1] - dst_bgr.data[1]).abs() < 1e-5);
    }

    // ── bgra_to_jzczhz ───────────────────────────────────────────────────────

    #[test]
    fn test_bgra_to_jzczhz_alpha_passthrough() {
        let src = make_src_u8(vec![0, 255, 0, 90], 1, 4); // BGR green + alpha
        let mut dst = make_dst_f32(1, 4);
        bgra_to_jzczhz(&src, &mut dst, DISPLAY_LUMINANCE, TransferFunction::Srgb).unwrap();
        let alpha = dst.data[3];
        assert!(
            (alpha - 90.0 / 255.0).abs() < 0.01,
            "Alpha ~0.353, got {alpha}"
        );
    }

    // ── jzazbz vs jzczhz consistency ─────────────────────────────────────────

    #[test]
    fn test_jzazbz_and_jzczhz_same_jz_for_same_input() {
        // Both share the same Jz lightness — must agree
        let src1 = make_src_u8(vec![200, 100, 50], 1, 3);
        let src2 = make_src_u8(vec![200, 100, 50], 1, 3);
        let mut dst_lab = make_dst_f32(1, 3);
        let mut dst_lch = make_dst_f32(1, 3);
        rgb_to_jzazbz(
            &src1,
            &mut dst_lab,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        rgb_to_jzczhz(
            &src2,
            &mut dst_lch,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert!(
            (dst_lab.data[0] - dst_lch.data[0]).abs() < 1e-4,
            "Jz must match between Jzazbz and Jzczhz: {} vs {}",
            dst_lab.data[0],
            dst_lch.data[0]
        );
    }

    #[test]
    fn test_jzazbz_chroma_equals_jzczhz_cz() {
        // Cz = sqrt(az² + bz²)
        let src1 = make_src_u8(vec![255, 0, 0], 1, 3);
        let src2 = make_src_u8(vec![255, 0, 0], 1, 3);
        let mut dst_lab = make_dst_f32(1, 3);
        let mut dst_lch = make_dst_f32(1, 3);
        rgb_to_jzazbz(
            &src1,
            &mut dst_lab,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        rgb_to_jzczhz(
            &src2,
            &mut dst_lch,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        let cz_from_lab = (dst_lab.data[1].powi(2) + dst_lab.data[2].powi(2)).sqrt();
        let cz_from_lch = dst_lch.data[1];
        assert!(
            (cz_from_lab - cz_from_lch).abs() < 1e-4,
            "Cz mismatch: sqrt(az²+bz²)={cz_from_lab} vs Cz={cz_from_lch}"
        );
    }

    // ── transfer function ─────────────────────────────────────────────────────

    #[test]
    fn test_linear_transfer_function_differs_from_srgb() {
        let src1 = make_src_u8(vec![128, 64, 32], 1, 3);
        let src2 = make_src_u8(vec![128, 64, 32], 1, 3);
        let mut dst_srgb = make_dst_f32(1, 3);
        let mut dst_linear = make_dst_f32(1, 3);
        rgb_to_jzazbz(
            &src1,
            &mut dst_srgb,
            DISPLAY_LUMINANCE,
            TransferFunction::Srgb,
        )
        .unwrap();
        rgb_to_jzazbz(
            &src2,
            &mut dst_linear,
            DISPLAY_LUMINANCE,
            TransferFunction::Linear,
        )
        .unwrap();
        assert_ne!(
            dst_srgb.data[0], dst_linear.data[0],
            "Srgb and Linear transfer functions should produce different Jz"
        );
    }
}
