/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::{ColorError, ImageBuffer, ImageBufferMut};

fn linear_to_gamma_channels<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    dst.validate()?;
    src.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut lut_table = [0u8; 256];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 255.0)) * 255.)
            .round()
            .min(255.) as u8;
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
            for (dst, &src) in dst.iter_mut().zip(src.iter()) {
                *dst = lut_table[src as usize];
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
                dst[3] = src[3];
            }
        }
    }

    Ok(())
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
pub fn linear_u8_to_rgb(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    linear_to_gamma_channels::<{ ImageConfiguration::Rgb as u8 }, false>(
        src,
        dst,
        transfer_function,
    )
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
pub fn linear_u8_to_rgba(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    linear_to_gamma_channels::<{ ImageConfiguration::Rgba as u8 }, true>(
        src,
        dst,
        transfer_function,
    )
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
pub fn linear_u8_to_bgra(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    linear_to_gamma_channels::<{ ImageConfiguration::Bgra as u8 }, true>(
        src,
        dst,
        transfer_function,
    )
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
pub fn linear_u8_to_bgr(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    linear_to_gamma_channels::<{ ImageConfiguration::Bgr as u8 }, false>(
        src,
        dst,
        transfer_function,
    )
}

#[cfg(test)]
mod tests_linear_u8_to_gamma {
    use super::*;
    use crate::BufferStore;

    fn make_src(data: Vec<u8>, width: u32, height: u32, channels: u32) -> ImageBuffer<'static, u8> {
        ImageBuffer::from_vec(data, width, height, channels).unwrap()
    }

    fn make_dst(width: u32, height: u32, channels: u32) -> ImageBufferMut<'static, u8> {
        ImageBufferMut::new(
            BufferStore::Owned(vec![0u8; (width * height * channels) as usize]),
            width,
            height,
            channels,
        )
        .unwrap()
    }

    // ── boundary values ───────────────────────────────────────────────────────

    #[test]
    fn rgb_zero_maps_to_zero() {
        let src = make_src(vec![0, 0, 0], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_u8_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[0, 0, 0]);
    }

    #[test]
    fn rgb_255_maps_to_255() {
        let src = make_src(vec![255, 255, 255], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_u8_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[255, 255, 255]);
    }

    #[test]
    fn bgr_zero_maps_to_zero() {
        let src = make_src(vec![0, 0, 0], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_u8_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[0, 0, 0]);
    }

    #[test]
    fn bgr_255_maps_to_255() {
        let src = make_src(vec![255, 255, 255], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_u8_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[255, 255, 255]);
    }

    // ── alpha passthrough (not gamma-encoded) ─────────────────────────────────

    #[test]
    fn rgba_alpha_is_not_transformed() {
        for alpha in [0u8, 64, 128, 192, 255] {
            let src = make_src(vec![128, 128, 128, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            linear_u8_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[3], alpha, "alpha={alpha} was modified");
        }
    }

    #[test]
    fn bgra_alpha_is_not_transformed() {
        for alpha in [0u8, 64, 128, 192, 255] {
            let src = make_src(vec![100, 100, 100, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            linear_u8_to_bgra(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[3], alpha, "alpha={alpha} was modified");
        }
    }

    #[test]
    fn rgba_alpha_zero_rgb_still_transformed() {
        let src = make_src(vec![128, 64, 32, 0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        linear_u8_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        // RGB channels must be gamma-encoded regardless of alpha
        let expected_r = (TransferFunction::Srgb.gamma(128.0 / 255.0) * 255.0).round() as u8;
        assert_eq!(d[0], expected_r, "R channel not transformed when alpha=0");
        assert_eq!(d[3], 0, "alpha must remain 0");
    }

    // ── lut correctness: gamma(x) = expected ─────────────────────────────────

    #[test]
    fn rgb_lut_matches_direct_gamma() {
        for input in (0u8..=255).step_by(16) {
            let expected = (TransferFunction::Srgb.gamma(input as f32 / 255.0) * 255.0)
                .round()
                .min(255.0) as u8;
            let src = make_src(vec![input, input, input], 1, 1, 3);
            let mut dst = make_dst(1, 1, 3);
            linear_u8_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[0], expected, "input={input}");
            assert_eq!(d[1], expected, "input={input}");
            assert_eq!(d[2], expected, "input={input}");
        }
    }

    #[test]
    fn bgr_lut_matches_direct_gamma() {
        for input in (0u8..=255).step_by(16) {
            let expected = (TransferFunction::Srgb.gamma(input as f32 / 255.0) * 255.0)
                .round()
                .min(255.0) as u8;
            let src = make_src(vec![input, input, input], 1, 1, 3);
            let mut dst = make_dst(1, 1, 3);
            linear_u8_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[0], expected, "input={input}");
        }
    }

    #[test]
    fn rgba_rgb_channels_lut_matches_direct_gamma() {
        for input in (0u8..=255).step_by(32) {
            let expected = (TransferFunction::Srgb.gamma(input as f32 / 255.0) * 255.0)
                .round()
                .min(255.0) as u8;
            let src = make_src(vec![input, input, input, 255], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            linear_u8_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[0], expected, "R input={input}");
            assert_eq!(d[1], expected, "G input={input}");
            assert_eq!(d[2], expected, "B input={input}");
        }
    }

    // ── rgb and bgr produce same output for grey pixels ───────────────────────

    #[test]
    fn rgb_bgr_agree_on_grey() {
        for v in (0u8..=255).step_by(32) {
            let src_rgb = make_src(vec![v, v, v], 1, 1, 3);
            let src_bgr = make_src(vec![v, v, v], 1, 1, 3);
            let mut dst_rgb = make_dst(1, 1, 3);
            let mut dst_bgr = make_dst(1, 1, 3);
            linear_u8_to_rgb(&src_rgb, &mut dst_rgb, TransferFunction::Srgb).unwrap();
            linear_u8_to_bgr(&src_bgr, &mut dst_bgr, TransferFunction::Srgb).unwrap();
            assert_eq!(&*dst_rgb.data.borrow(), &*dst_bgr.data.borrow(), "v={v}");
        }
    }

    #[test]
    fn rgba_bgra_agree_on_grey() {
        for v in (0u8..=255).step_by(32) {
            let src_rgba = make_src(vec![v, v, v, 200], 1, 1, 4);
            let src_bgra = make_src(vec![v, v, v, 200], 1, 1, 4);
            let mut dst_rgba = make_dst(1, 1, 4);
            let mut dst_bgra = make_dst(1, 1, 4);
            linear_u8_to_rgba(&src_rgba, &mut dst_rgba, TransferFunction::Srgb).unwrap();
            linear_u8_to_bgra(&src_bgra, &mut dst_bgra, TransferFunction::Srgb).unwrap();
            assert_eq!(&*dst_rgba.data.borrow(), &*dst_bgra.data.borrow(), "v={v}");
        }
    }

    // ── linear transfer function is identity ──────────────────────────────────

    #[test]
    fn linear_transfer_rgb_is_identity() {
        let data: Vec<u8> = (0u8..=255).step_by(4).flat_map(|v| [v, v, v]).collect();
        let px = data.len() as u32 / 3;
        let src = make_src(data.clone(), px, 1, 3);
        let mut dst = make_dst(px, 1, 3);
        linear_u8_to_rgb(&src, &mut dst, TransferFunction::Linear).unwrap();
        assert_eq!(&*dst.data.borrow(), data.as_slice());
    }

    #[test]
    fn linear_transfer_rgba_rgb_channels_identity() {
        let data: Vec<u8> = (0u8..=255)
            .step_by(16)
            .flat_map(|v| [v, v, v, 128u8])
            .collect();
        let px = data.len() as u32 / 4;
        let src = make_src(data.clone(), px, 1, 4);
        let mut dst = make_dst(px, 1, 4);
        linear_u8_to_rgba(&src, &mut dst, TransferFunction::Linear).unwrap();
        let d = dst.data.borrow();
        for (i, chunk) in d.chunks(4).enumerate() {
            let orig = data[i * 4];
            assert_eq!(chunk[0], orig, "R px={i}");
            assert_eq!(chunk[1], orig, "G px={i}");
            assert_eq!(chunk[2], orig, "B px={i}");
            assert_eq!(chunk[3], 128, "A px={i}");
        }
    }

    // ── monotonicity ──────────────────────────────────────────────────────────

    #[test]
    fn rgb_output_monotonically_non_decreasing() {
        let data: Vec<u8> = (0u8..=255).flat_map(|v| [v, v, v]).collect();
        let src = make_src(data, 256, 1, 3);
        let mut dst = make_dst(256, 1, 3);
        linear_u8_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        // check only the R channel (all equal anyway)
        let reds: Vec<u8> = d.chunks(3).map(|c| c[0]).collect();
        for i in 1..reds.len() {
            assert!(
                reds[i] >= reds[i - 1],
                "non-monotone at index {i}: {} < {}",
                reds[i],
                reds[i - 1]
            );
        }
    }

    // ── multi-pixel / multi-row consistency ───────────────────────────────────

    #[test]
    fn rgb_multi_pixel_consistent() {
        let src = make_src(vec![100, 150, 200].repeat(6), 6, 1, 3);
        let mut dst = make_dst(6, 1, 3);
        linear_u8_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn rgb_multi_row_consistent() {
        let src = make_src(vec![80, 160, 240].repeat(4), 2, 2, 3);
        let mut dst = make_dst(2, 2, 3);
        linear_u8_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn rgba_multi_pixel_consistent() {
        let src = make_src(vec![100, 150, 200, 128].repeat(4), 4, 1, 4);
        let mut dst = make_dst(4, 1, 4);
        linear_u8_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2], d[3]];
        for (i, chunk) in d.chunks(4).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    // ── channels mismatch returns error ───────────────────────────────────────

    #[test]
    fn rgb_rejects_wrong_channel_count() {
        // 4-channel src passed to rgb (expects 3)
        let src = make_src(vec![0u8; 4], 1, 1, 4);
        let mut dst = make_dst(1, 1, 3);
        assert!(linear_u8_to_rgb(&src, &mut dst, TransferFunction::Srgb).is_err());
    }

    #[test]
    fn rgba_rejects_wrong_channel_count() {
        let src = make_src(vec![0u8; 3], 1, 1, 3);
        let mut dst = make_dst(1, 1, 4);
        assert!(linear_u8_to_rgba(&src, &mut dst, TransferFunction::Srgb).is_err());
    }
}
