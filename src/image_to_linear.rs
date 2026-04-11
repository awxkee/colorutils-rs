/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::{ColorError, ImageBuffer, ImageBufferMut};

#[allow(clippy::type_complexity)]
fn channels_to_linear<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    src.validate()?;
    dst.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut lut_table = [0f32; 256];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = transfer_function.linearize(i as f32 * (1. / 255.0));
    }

    let mut a_lut_table = [0.; 256];
    for (i, lut) in a_lut_table.iter_mut().enumerate() {
        *lut = i as f32 * (1. / 255.0);
    }

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;

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
            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                dst[0] = lut_table[src[0] as usize];
                dst[1] = lut_table[src[1] as usize];
                dst[2] = lut_table[src[2] as usize];
            }
        } else if image_configuration == ImageConfiguration::Bgra
            || image_configuration == ImageConfiguration::Rgba
        {
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                dst[0] = lut_table[src[0] as usize];
                dst[1] = lut_table[src[1] as usize];
                dst[2] = lut_table[src[2] as usize];
                dst[3] = a_lut_table[src[3] as usize];
            }
        }
    }

    Ok(())
}

/// This function converts RGB to linear colorspace
///
/// This function converts RGB to linear color space. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_linear(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear::<{ ImageConfiguration::Rgb as u8 }, false>(src, dst, transfer_function)
}

/// This function converts RGBA to liner color space
///
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
pub fn rgba_to_linear(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear::<{ ImageConfiguration::Rgba as u8 }, true>(src, dst, transfer_function)
}

/// This function converts BGRA to Linear.
///
/// This function converts BGRA to Linear, alpha channel is normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgra_to_linear(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear::<{ ImageConfiguration::Bgra as u8 }, true>(src, dst, transfer_function)
}

/// This function converts BGR to linear
///
/// This function converts BGR to linear color space. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgr_to_linear(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear::<{ ImageConfiguration::Bgr as u8 }, false>(src, dst, transfer_function)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BufferStore;

    fn make_src(width: u32, height: u32, channels: u32, data: Vec<u8>) -> ImageBuffer<'static, u8> {
        ImageBuffer {
            width,
            height,
            channels,
            data: data.into(),
            stride: width * channels,
        }
    }

    fn make_dst(width: u32, height: u32, channels: u32) -> ImageBufferMut<'static, f32> {
        let len = (width * height * channels) as usize;
        ImageBufferMut {
            width,
            height,
            channels,
            data: BufferStore::Owned(vec![0f32; len]),
            stride: width * channels,
        }
    }

    #[test]
    fn alpha_channel_is_linear_not_gamma() {
        // Alpha at 128 → should be ~0.502 (128/255), not sRGB-linearised (~0.216)
        let src = make_src(1, 1, 4, vec![0u8, 0, 0, 128]);
        let mut dst = make_dst(1, 1, 4);
        let _ = rgba_to_linear(&src, &mut dst, TransferFunction::Srgb);
        let alpha_out = dst.data.borrow()[3];
        let expected = 128.0 / 255.0;
        assert!(
            (alpha_out - expected).abs() < 1e-4,
            "expected alpha ≈ {expected}, got {alpha_out}"
        );
    }

    #[test]
    fn color_channel_is_gamma_corrected_not_linear() {
        // 128 in sRGB ≠ 128/255 ≈ 0.502 in linear; it should be ≈ 0.216
        let src = make_src(1, 1, 3, vec![128u8, 128, 128]);
        let mut dst = make_dst(1, 1, 3);
        let _ = rgb_to_linear(&src, &mut dst, TransferFunction::Srgb);
        let r = dst.data.borrow()[0];
        // sRGB linearisation of 0.502 ≈ 0.2158
        assert!(
            (r - 0.2158).abs() < 5e-3,
            "expected r ≈ 0.2158 (sRGB linearised), got {r}"
        );
    }

    #[test]
    fn black_pixel_maps_to_zero() {
        let src = make_src(1, 1, 3, vec![0u8, 0, 0]);
        let mut dst = make_dst(1, 1, 3);
        let _ = rgb_to_linear(&src, &mut dst, TransferFunction::Srgb);
        for v in dst.data.borrow().iter() {
            assert_eq!(*v, 0.0, "black should map to 0.0");
        }
    }

    #[test]
    fn white_pixel_maps_to_one() {
        let src = make_src(1, 1, 3, vec![255u8, 255, 255]);
        let mut dst = make_dst(1, 1, 3);
        let _ = rgb_to_linear(&src, &mut dst, TransferFunction::Srgb);
        for v in dst.data.borrow().iter() {
            assert!((v - 1.0).abs() < 1e-4, "white should map to ≈ 1.0, got {v}");
        }
    }

    // ── channel-order: BGR vs RGB ─────────────────────────────────────────────

    #[test]
    fn bgr_channel_order_is_preserved() {
        // A pixel with distinct R/G/B values; we verify the values land in
        // the right output slots regardless of byte ordering.
        let r_in = 200u8;
        let g_in = 100u8;
        let b_in = 50u8;

        let src_rgb = make_src(1, 1, 3, vec![r_in, g_in, b_in]);
        let mut dst_rgb = make_dst(1, 1, 3);
        let _ = rgb_to_linear(&src_rgb, &mut dst_rgb, TransferFunction::Srgb);

        let src_bgr = make_src(1, 1, 3, vec![b_in, g_in, r_in]);
        let mut dst_bgr = make_dst(1, 1, 3);
        let _ = bgr_to_linear(&src_bgr, &mut dst_bgr, TransferFunction::Srgb);

        let rgb = dst_rgb.data.borrow();
        let bgr = dst_bgr.data.borrow();
        // bgr output slot 0 = Blue, rgb output slot 2 = Blue
        assert!((bgr[0] - rgb[2]).abs() < 1e-5, "B mismatch");
        assert!((bgr[1] - rgb[1]).abs() < 1e-5, "G mismatch");
        assert!((bgr[2] - rgb[0]).abs() < 1e-5, "R mismatch");
    }

    // ── multi-pixel / stride correctness ─────────────────────────────────────

    #[test]
    fn multi_pixel_row_all_channels_converted() {
        // 3 pixels in a single row
        let src = make_src(
            3,
            1,
            3,
            vec![
                255, 0, 0, // red
                0, 255, 0, // green
                0, 0, 255, // blue
            ],
        );
        let mut dst = make_dst(3, 1, 3);
        let _ = rgb_to_linear(&src, &mut dst, TransferFunction::Srgb);
        let d = dst.data.borrow();
        assert!((d[0] - 1.0).abs() < 1e-4, "pixel 0 R should be 1.0");
        assert_eq!(d[1], 0.0, "pixel 0 G should be 0.0");
        assert_eq!(d[2], 0.0, "pixel 0 B should be 0.0");
        assert_eq!(d[3], 0.0, "pixel 1 R should be 0.0");
        assert!((d[4] - 1.0).abs() < 1e-4, "pixel 1 G should be 1.0");
    }

    #[test]
    fn multi_row_image_respects_stride() {
        // 1×2 image (two rows of a single pixel)
        let src = make_src(
            1,
            2,
            3,
            vec![
                255, 255, 255, // row 0: white
                0, 0, 0, // row 1: black
            ],
        );
        let mut dst = make_dst(1, 2, 3);
        let _ = rgb_to_linear(&src, &mut dst, TransferFunction::Srgb);
        let d = dst.data.borrow();
        assert!((d[0] - 1.0).abs() < 1e-4, "row 0 should be white (1.0)");
        assert_eq!(d[3], 0.0, "row 1 should be black (0.0)");
    }

    #[test]
    fn rgba_alpha_fully_opaque_remains_one() {
        let src = make_src(1, 1, 4, vec![128, 64, 32, 255]);
        let mut dst = make_dst(1, 1, 4);
        let _ = rgba_to_linear(&src, &mut dst, TransferFunction::Srgb);
        let alpha = dst.data.borrow()[3];
        assert!((alpha - 1.0).abs() < 1e-4, "alpha 255 → 1.0, got {alpha}");
    }

    #[test]
    fn rgba_alpha_fully_transparent_is_zero() {
        let src = make_src(1, 1, 4, vec![255, 255, 255, 0]);
        let mut dst = make_dst(1, 1, 4);
        let _ = rgba_to_linear(&src, &mut dst, TransferFunction::Srgb);
        let alpha = dst.data.borrow()[3];
        assert_eq!(alpha, 0.0, "alpha 0 → 0.0");
    }

    #[test]
    fn mismatched_dimensions_returns_error() {
        let src = make_src(2, 2, 3, vec![0u8; 12]);
        let mut dst = make_dst(4, 4, 3); // different size
        let result = rgb_to_linear(&src, &mut dst, TransferFunction::Srgb);
        assert!(result.is_err(), "expected error for dimension mismatch");
    }

    #[test]
    fn linear_transfer_function_is_identity() {
        // With a linear (gamma = 1.0) transfer function 128 → 128/255 ≈ 0.502
        let src = make_src(1, 1, 3, vec![128, 128, 128]);
        let mut dst = make_dst(1, 1, 3);
        let _ = rgb_to_linear(&src, &mut dst, TransferFunction::Linear);
        let v = dst.data.borrow()[0];
        let expected = 128.0 / 255.0;
        assert!(
            (v - expected).abs() < 1e-4,
            "linear TF: expected {expected}, got {v}"
        );
    }
}
