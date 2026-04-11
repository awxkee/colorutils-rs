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
    src: &ImageBuffer<'_, f32>,
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

    let mut lut_table = [0u8; 2049];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.)
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
                let r = (src[image_configuration.r_index()] * 2048.).round() as usize;
                let g = (src[image_configuration.g_index()] * 2048.).round() as usize;
                let b = (src[image_configuration.b_index()] * 2048.).round() as usize;
                dst[image_configuration.r_index()] = lut_table[r.min(2048)];
                dst[image_configuration.g_index()] = lut_table[g.min(2048)];
                dst[image_configuration.b_index()] = lut_table[b.min(2048)];
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
                let r = (src[image_configuration.r_index()] * 2048.).round() as usize;
                let g = (src[image_configuration.g_index()] * 2048.).round() as usize;
                let b = (src[image_configuration.b_index()] * 2048.).round() as usize;
                dst[image_configuration.r_index()] = lut_table[r.min(2048)];
                dst[image_configuration.g_index()] = lut_table[g.min(2048)];
                dst[image_configuration.b_index()] = lut_table[b.min(2048)];
                dst[image_configuration.a_index()] = (src[image_configuration.a_index()] * 255.)
                    .round()
                    .min(255.) as u8;
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
pub fn linear_to_rgb(
    src: &ImageBuffer<'_, f32>,
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
pub fn linear_to_rgba(
    src: &ImageBuffer<'_, f32>,
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
pub fn linear_to_bgra(
    src: &ImageBuffer<'_, f32>,
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
pub fn linear_to_bgr(
    src: &ImageBuffer<'_, f32>,
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
mod tests_linear_f32_to_gamma {
    use super::*;
    use crate::BufferStore;

    fn make_src(
        data: Vec<f32>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> ImageBuffer<'static, f32> {
        ImageBuffer::from_vec(data, width, height, width * channels, channels).unwrap()
    }

    fn make_dst(width: u32, height: u32, channels: u32) -> ImageBufferMut<'static, u8> {
        ImageBufferMut::new(
            BufferStore::Owned(vec![0u8; (width * height * channels) as usize]),
            width,
            height,
            width * channels,
            channels,
        )
        .unwrap()
    }

    fn lut_encode(v: f32) -> u8 {
        let idx = (v.clamp(0., 1.) * 2048.).round() as usize;
        let idx = idx.min(2048);
        (TransferFunction::Srgb.gamma(idx as f32 / 2048.0) * 255.)
            .round()
            .min(255.) as u8
    }

    fn assert_approx(a: u8, b: u8, label: &str) {
        assert!(
            (a as i16 - b as i16).abs() <= 1,
            "{label}: got {a}, expected {b}"
        );
    }

    // ── boundary values ───────────────────────────────────────────────────────

    #[test]
    fn rgb_zero_maps_to_zero() {
        let src = make_src(vec![0.0, 0.0, 0.0], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[0, 0, 0]);
    }

    #[test]
    fn rgb_one_maps_to_255() {
        let src = make_src(vec![1.0, 1.0, 1.0], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[255, 255, 255]);
    }

    #[test]
    fn bgr_zero_maps_to_zero() {
        let src = make_src(vec![0.0, 0.0, 0.0], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[0, 0, 0]);
    }

    #[test]
    fn bgr_one_maps_to_255() {
        let src = make_src(vec![1.0, 1.0, 1.0], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst.data.borrow(), &[255, 255, 255]);
    }

    // ── clamping ──────────────────────────────────────────────────────────────

    #[test]
    fn rgb_above_one_clamps_to_255() {
        for v in [1.001f32, 1.5, 2.0, 100.0] {
            let src = make_src(vec![v, v, v], 1, 1, 3);
            let mut dst = make_dst(1, 1, 3);
            linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
            assert_eq!(&*dst.data.borrow(), &[255, 255, 255], "v={v}");
        }
    }

    #[test]
    fn rgb_below_zero_clamps_to_zero() {
        for v in [-0.001f32, -1.0, -100.0] {
            let src = make_src(vec![v, v, v], 1, 1, 3);
            let mut dst = make_dst(1, 1, 3);
            linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
            assert_eq!(&*dst.data.borrow(), &[0, 0, 0], "v={v}");
        }
    }

    // ── alpha passthrough ─────────────────────────────────────────────────────

    #[test]
    fn rgba_alpha_is_denormalized() {
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src(vec![0.5, 0.5, 0.5, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            linear_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            let expected_a = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected_a, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn bgra_alpha_is_denormalized() {
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src(vec![0.5, 0.5, 0.5, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            linear_to_bgra(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            let expected_a = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected_a, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn rgba_rgb_channels_encoded_regardless_of_alpha() {
        let src = make_src(vec![0.5, 0.3, 0.1, 0.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        linear_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_ne!(d[0], 0, "R must be gamma-encoded even when alpha=0");
        assert_eq!(d[3], 0, "alpha must be 0");
    }

    // ── lut correctness ───────────────────────────────────────────────────────

    #[test]
    fn rgb_lut_matches_expected_encoding() {
        for v in [0.0f32, 0.1, 0.2, 0.5, 0.8, 1.0] {
            let expected = lut_encode(v);
            let src = make_src(vec![v, v, v], 1, 1, 3);
            let mut dst = make_dst(1, 1, 3);
            linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            assert_approx(d[0], expected, &format!("R v={v}"));
            assert_approx(d[1], expected, &format!("G v={v}"));
            assert_approx(d[2], expected, &format!("B v={v}"));
        }
    }

    #[test]
    fn bgr_lut_matches_expected_encoding() {
        for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let expected = lut_encode(v);
            let src = make_src(vec![v, v, v], 1, 1, 3);
            let mut dst = make_dst(1, 1, 3);
            linear_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            assert_approx(d[0], expected, &format!("v={v}"));
        }
    }

    #[test]
    fn rgba_rgb_lut_matches_expected_encoding() {
        for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let expected = lut_encode(v);
            let src = make_src(vec![v, v, v, 1.0], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            linear_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            assert_approx(d[0], expected, &format!("R v={v}"));
            assert_approx(d[1], expected, &format!("G v={v}"));
            assert_approx(d[2], expected, &format!("B v={v}"));
        }
    }

    // ── channel ordering: RGB vs BGR ──────────────────────────────────────────

    #[test]
    fn rgb_bgr_agree_on_grey() {
        for v in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
            let src_rgb = make_src(vec![v, v, v], 1, 1, 3);
            let src_bgr = make_src(vec![v, v, v], 1, 1, 3);
            let mut dst_rgb = make_dst(1, 1, 3);
            let mut dst_bgr = make_dst(1, 1, 3);
            linear_to_rgb(&src_rgb, &mut dst_rgb, TransferFunction::Srgb).unwrap();
            linear_to_bgr(&src_bgr, &mut dst_bgr, TransferFunction::Srgb).unwrap();
            assert_eq!(&*dst_rgb.data.borrow(), &*dst_bgr.data.borrow(), "v={v}");
        }
    }

    #[test]
    fn rgba_bgra_agree_on_grey() {
        for v in [0.2f32, 0.5, 0.8] {
            let src_rgba = make_src(vec![v, v, v, 0.6], 1, 1, 4);
            let src_bgra = make_src(vec![v, v, v, 0.6], 1, 1, 4);
            let mut dst_rgba = make_dst(1, 1, 4);
            let mut dst_bgra = make_dst(1, 1, 4);
            linear_to_rgba(&src_rgba, &mut dst_rgba, TransferFunction::Srgb).unwrap();
            linear_to_bgra(&src_bgra, &mut dst_bgra, TransferFunction::Srgb).unwrap();
            assert_eq!(&*dst_rgba.data.borrow(), &*dst_bgra.data.borrow(), "v={v}");
        }
    }

    #[test]
    fn rgb_distinct_channels_ordered_correctly() {
        // R=1.0 G=0.0 B=0.5 — output must respect channel positions
        let src = make_src(vec![1.0, 0.0, 0.5], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_eq!(d[0], 255, "R");
        assert_eq!(d[1], 0, "G");
        assert_approx(d[2], lut_encode(0.5), "B");
    }

    #[test]
    fn bgr_distinct_channels_ordered_correctly() {
        // BGR layout: first byte = B, second = G, third = R
        let src = make_src(vec![0.0, 0.5, 1.0], 1, 1, 3); // B=0, G=0.5, R=1.0
        let mut dst = make_dst(1, 1, 3);
        linear_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_eq!(d[0], 0, "B");
        assert_approx(d[1], lut_encode(0.5), "G");
        assert_eq!(d[2], 255, "R");
    }

    // ── linear transfer is identity ───────────────────────────────────────────

    #[test]
    fn linear_transfer_rgb_is_identity() {
        for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let expected = (v * 255.).round() as u8;
            let src = make_src(vec![v, v, v], 1, 1, 3);
            let mut dst = make_dst(1, 1, 3);
            linear_to_rgb(&src, &mut dst, TransferFunction::Linear).unwrap();
            let d = dst.data.borrow();
            assert_approx(d[0], expected, &format!("v={v}"));
        }
    }

    #[test]
    fn linear_transfer_rgba_rgb_is_identity() {
        for v in [0.0f32, 0.5, 1.0] {
            let expected = (v * 255.).round() as u8;
            let src = make_src(vec![v, v, v, 0.5], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            linear_to_rgba(&src, &mut dst, TransferFunction::Linear).unwrap();
            let d = dst.data.borrow();
            assert_approx(d[0], expected, &format!("R v={v}"));
            assert_approx(d[1], expected, &format!("G v={v}"));
            assert_approx(d[2], expected, &format!("B v={v}"));
        }
    }

    // ── monotonicity ──────────────────────────────────────────────────────────

    #[test]
    fn rgb_output_monotonically_non_decreasing() {
        let steps = 256usize;
        let data: Vec<f32> = (0..steps)
            .flat_map(|i| {
                let v = i as f32 / (steps - 1) as f32;
                [v, v, v]
            })
            .collect();
        let src = make_src(data, steps as u32, 1, 3);
        let mut dst = make_dst(steps as u32, 1, 3);
        linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let reds: Vec<u8> = d.chunks(3).map(|c| c[0]).collect();
        for i in 1..reds.len() {
            assert!(
                reds[i] >= reds[i - 1],
                "non-monotone at {i}: {} < {}",
                reds[i],
                reds[i - 1]
            );
        }
    }

    // ── multi-pixel / multi-row consistency ───────────────────────────────────

    #[test]
    fn rgb_multi_pixel_consistent() {
        let src = make_src(vec![0.4, 0.6, 0.8f32].repeat(6), 6, 1, 3);
        let mut dst = make_dst(6, 1, 3);
        linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn rgba_multi_pixel_consistent() {
        let src = make_src(vec![0.3, 0.5, 0.7, 0.9f32].repeat(4), 4, 1, 4);
        let mut dst = make_dst(4, 1, 4);
        linear_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2], d[3]];
        for (i, chunk) in d.chunks(4).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn rgb_multi_row_consistent() {
        let src = make_src(vec![0.2, 0.4, 0.6f32].repeat(4), 2, 2, 3);
        let mut dst = make_dst(2, 2, 3);
        linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    // ── error paths ───────────────────────────────────────────────────────────

    #[test]
    fn rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 4], 1, 1, 4, 4).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(linear_to_rgb(&src, &mut dst, TransferFunction::Srgb).is_err());
    }

    #[test]
    fn rgba_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 3], 1, 1, 3, 3).unwrap();
        let mut dst = make_dst(1, 1, 4);
        assert!(linear_to_rgba(&src, &mut dst, TransferFunction::Srgb).is_err());
    }
}
