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
    dst: &mut ImageBufferMut<'_, u8>,
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

    let mut lut_table = [0u8; 256];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = (transfer_function.linearize(i as f32 * (1. / 255.0)) * 255.)
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
                dst[3] = src[3];
            }
        }
    }
    Ok(())
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
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear::<{ ImageConfiguration::Rgb as u8 }, false>(src, dst, transfer_function)
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
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear::<{ ImageConfiguration::Rgba as u8 }, true>(src, dst, transfer_function)
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
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear::<{ ImageConfiguration::Bgra as u8 }, true>(src, dst, transfer_function)
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
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear::<{ ImageConfiguration::Bgr as u8 }, false>(src, dst, transfer_function)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BufferStore;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_src(width: u32, height: u32, channels: u32, data: Vec<u8>) -> ImageBuffer<'static, u8> {
        ImageBuffer {
            width,
            height,
            channels,
            data: data.into(),
            stride: width * channels,
        }
    }

    fn make_dst(width: u32, height: u32, channels: u32) -> ImageBufferMut<'static, u8> {
        let len = (width * height * channels) as usize;
        ImageBufferMut {
            width,
            height,
            channels,
            data: BufferStore::Owned(vec![0u8; len]),
            stride: width * channels,
        }
    }

    // Helper: linearise a single byte through the sRGB transfer function,
    // matching exactly how the LUT is built in `channels_to_linear`.
    fn srgb_linearize_u8(v: u8) -> u8 {
        (TransferFunction::Srgb.linearize(v as f32 / 255.0) * 255.0).min(255.0) as u8
    }

    // ── validation ────────────────────────────────────────────────────────────

    #[test]
    fn unsupported_channel_count_returns_error() {
        let src = make_src(1, 1, 2, vec![128u8, 64]);
        let mut dst = make_dst(1, 1, 2);
        let result = rgb_to_linear_u8(&src, &mut dst, TransferFunction::Srgb);
        assert!(matches!(
            result,
            Err(ColorError::UnsupportedChannelsCount(2))
        ));
    }

    #[test]
    fn mismatched_dimensions_returns_error() {
        let src = make_src(2, 2, 3, vec![0u8; 12]);
        let mut dst = make_dst(4, 4, 3);
        let result = rgb_to_linear_u8(&src, &mut dst, TransferFunction::Srgb);
        assert!(result.is_err());
    }

    // ── black / white anchors ─────────────────────────────────────────────────

    #[test]
    fn black_maps_to_zero() {
        let src = make_src(1, 1, 3, vec![0, 0, 0]);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[0u8, 0, 0]);
    }

    #[test]
    fn white_maps_to_255() {
        let src = make_src(1, 1, 3, vec![255, 255, 255]);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[255u8, 255, 255]);
    }

    // ── LUT correctness ───────────────────────────────────────────────────────

    #[test]
    fn mid_grey_matches_srgb_lut() {
        let v = 128u8;
        let src = make_src(1, 1, 3, vec![v, v, v]);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let expected = srgb_linearize_u8(v);
        let d = dst.data.borrow();
        assert_eq!(d[0], expected, "R: expected {expected}, got {}", d[0]);
        assert_eq!(d[1], expected, "G: expected {expected}, got {}", d[1]);
        assert_eq!(d[2], expected, "B: expected {expected}, got {}", d[2]);
    }

    #[test]
    fn all_256_lut_entries_are_monotonically_non_decreasing() {
        // Linearisation must be monotone — a brighter input can never produce
        // a darker output.
        for v in 1u8..=255 {
            let lo = srgb_linearize_u8(v - 1);
            let hi = srgb_linearize_u8(v);
            assert!(
                hi >= lo,
                "LUT not monotone: lut[{}]={} > lut[{}]={}",
                v,
                hi,
                v - 1,
                lo
            );
        }
    }

    #[test]
    fn linear_transfer_function_is_identity() {
        // With a γ=1 (linear) transfer function the LUT should be the identity.
        for v in 0u8..=255 {
            let out =
                (TransferFunction::Linear.linearize(v as f32 / 255.0) * 255.0).min(255.0) as u8;
            assert_eq!(out, v, "linear TF: lut[{v}]={out}, expected {v}");
        }
    }

    // ── sRGB vs linear TF produce different results for mid-grey ─────────────

    #[test]
    fn srgb_and_linear_tf_differ_for_mid_grey() {
        let src_srgb = make_src(1, 1, 3, vec![128, 128, 128]);
        let src_linear = make_src(1, 1, 3, vec![128, 128, 128]);
        let mut dst_srgb = make_dst(1, 1, 3);
        let mut dst_linear = make_dst(1, 1, 3);
        rgb_to_linear_u8(&src_srgb, &mut dst_srgb, TransferFunction::Srgb).unwrap();
        rgb_to_linear_u8(&src_linear, &mut dst_linear, TransferFunction::Linear).unwrap();
        let srgb_out = dst_srgb.data.borrow()[0];
        let linear_out = dst_linear.data.borrow()[0];
        assert_ne!(
            srgb_out, linear_out,
            "sRGB and linear TF should give different results for mid-grey"
        );
        assert_eq!(linear_out, 128, "linear TF should be identity for 128");
    }

    // ── RGBA: alpha is passed through unchanged ───────────────────────────────

    #[test]
    fn rgba_alpha_is_passed_through_unchanged() {
        for alpha in [0u8, 1, 127, 128, 254, 255] {
            let src = make_src(1, 1, 4, vec![200, 100, 50, alpha]);
            let mut dst = make_dst(1, 1, 4);
            rgba_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let out_alpha = dst.data.borrow()[3];
            assert_eq!(
                out_alpha, alpha,
                "alpha {alpha} should be passed through unchanged, got {out_alpha}"
            );
        }
    }

    #[test]
    fn rgba_colour_channels_are_linearised_independently_of_alpha() {
        let v = 200u8;
        let expected = srgb_linearize_u8(v);

        let src_opaque = make_src(1, 1, 4, vec![v, v, v, 255]);
        let src_transp = make_src(1, 1, 4, vec![v, v, v, 0]);
        let mut dst_opaque = make_dst(1, 1, 4);
        let mut dst_transp = make_dst(1, 1, 4);
        rgba_to_linear_u8(&src_opaque, &mut dst_opaque, TransferFunction::Srgb).unwrap();
        rgba_to_linear_u8(&src_transp, &mut dst_transp, TransferFunction::Srgb).unwrap();

        for i in 0..3 {
            assert_eq!(dst_opaque.data.borrow()[i], expected);
            assert_eq!(dst_transp.data.borrow()[i], expected);
        }
    }

    // ── BGRA: alpha is passed through unchanged ───────────────────────────────

    #[test]
    fn bgra_alpha_is_passed_through_unchanged() {
        let src = make_src(1, 1, 4, vec![50, 100, 200, 128]);
        let mut dst = make_dst(1, 1, 4);
        bgra_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(
            dst.data.borrow()[3],
            128,
            "BGRA alpha must be copied verbatim"
        );
    }

    // ── BGR / RGB channel ordering ────────────────────────────────────────────

    #[test]
    fn bgr_and_rgb_agree_on_grey() {
        // Grey is order-invariant: both paths must produce the same output.
        let src_rgb = make_src(1, 1, 3, vec![128, 128, 128]);
        let src_bgr = make_src(1, 1, 3, vec![128, 128, 128]);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        rgb_to_linear_u8(&src_rgb, &mut dst_rgb, TransferFunction::Srgb).unwrap();
        bgr_to_linear_u8(&src_bgr, &mut dst_bgr, TransferFunction::Srgb).unwrap();
        assert_eq!(
            &*dst_rgb.data.borrow(),
            &*dst_bgr.data.borrow(),
            "grey should give identical output for rgb and bgr"
        );
    }

    #[test]
    fn bgr_reversal_matches_rgb_with_reversed_input() {
        let (r, g, b) = (200u8, 128u8, 50u8);

        let src_rgb = make_src(1, 1, 3, vec![r, g, b]);
        let src_bgr = make_src(1, 1, 3, vec![b, g, r]); // reversed
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        rgb_to_linear_u8(&src_rgb, &mut dst_rgb, TransferFunction::Srgb).unwrap();
        bgr_to_linear_u8(&src_bgr, &mut dst_bgr, TransferFunction::Srgb).unwrap();

        // BGR output slot 0 is B_linear; RGB output slot 2 is B_linear
        let r_out = dst_rgb.data.borrow();
        let b_out = dst_bgr.data.borrow();
        assert_eq!(b_out[0], r_out[2], "B channel mismatch");
        assert_eq!(b_out[1], r_out[1], "G channel mismatch");
        assert_eq!(b_out[2], r_out[0], "R channel mismatch");
    }

    // ── multi-pixel / stride ──────────────────────────────────────────────────

    #[test]
    fn multi_pixel_row_each_pixel_converted_independently() {
        let src = make_src(
            3,
            1,
            3,
            vec![
                0, 0, 0, // black
                128, 128, 128, // mid-grey
                255, 255, 255, // white
            ],
        );
        let mut dst = make_dst(3, 1, 3);
        rgb_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_eq!(d[0], 0, "pixel 0 should be 0");
        assert_eq!(
            d[3],
            srgb_linearize_u8(128),
            "pixel 1 should be linearised 128"
        );
        assert_eq!(d[6], 255, "pixel 2 should be 255");
    }

    #[test]
    fn multi_row_image_respects_stride() {
        // 1×3 image: row 0 black, row 1 mid-grey, row 2 white
        let src = make_src(1, 3, 3, vec![0, 0, 0, 128, 128, 128, 255, 255, 255]);
        let mut dst = make_dst(1, 3, 3);
        rgb_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_eq!(d[0], 0, "row 0 should be 0");
        assert_eq!(
            d[3],
            srgb_linearize_u8(128),
            "row 1 should be linearised 128"
        );
        assert_eq!(d[6], 255, "row 2 should be 255");
    }

    // ── each channel is converted independently ───────────────────────────────

    #[test]
    fn each_rgb_channel_is_converted_independently() {
        let (r, g, b) = (200u8, 100u8, 50u8);
        let src = make_src(1, 1, 3, vec![r, g, b]);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_eq!(d[0], srgb_linearize_u8(r), "R channel");
        assert_eq!(d[1], srgb_linearize_u8(g), "G channel");
        assert_eq!(d[2], srgb_linearize_u8(b), "B channel");
    }

    #[test]
    fn each_rgba_colour_channel_is_converted_independently() {
        let (r, g, b, a) = (200u8, 100u8, 50u8, 180u8);
        let src = make_src(1, 1, 4, vec![r, g, b, a]);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_linear_u8(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_eq!(d[0], srgb_linearize_u8(r), "R channel");
        assert_eq!(d[1], srgb_linearize_u8(g), "G channel");
        assert_eq!(d[2], srgb_linearize_u8(b), "B channel");
        assert_eq!(d[3], a, "alpha passthrough");
    }
}
