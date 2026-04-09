/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_image_to_sigmoidal_row;

use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_image_to_sigmoidal;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_image_to_sigmoidal_row;
use crate::{ColorError, ImageBuffer, ImageBufferMut, Rgb};

#[allow(clippy::type_complexity)]
fn image_to_sigmoidal<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
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

    let mut _wide_row_handler: Option<unsafe fn(usize, &[u8], &mut [f32], u32) -> usize> = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler = Some(neon_image_to_sigmoidal::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler = Some(sse_image_to_sigmoidal_row::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handler = Some(avx_image_to_sigmoidal_row::<CHANNELS_CONFIGURATION, USE_ALPHA>);
    }

    const COLOR_SCALE: f32 = 1f32 / 255f32;

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;
    let width = src.width;

    for (dst, src) in dst
        .data
        .borrow_mut()
        .chunks_mut(dst_stride)
        .zip(src.data.chunks(src.stride()))
    {
        let src = &src[..src_r_width as usize];
        let dst = &mut dst[..dst_r_width as usize];

        let mut cx = 0usize;
        if let Some(dispatcher) = _wide_row_handler {
            cx = unsafe { dispatcher(cx, src, dst, width) };
        }

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            let dst = &mut dst[cx * 3..];
            let src = &src[cx * 3..];
            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                let rgb = Rgb::<u8>::new(
                    src[image_configuration.r_index()],
                    src[image_configuration.g_index()],
                    src[image_configuration.b_index()],
                );

                let sigmoidal = rgb.to_sigmoidal();

                dst[0] = sigmoidal.sr;
                dst[1] = sigmoidal.sg;
                dst[2] = sigmoidal.sb;
            }
        } else if image_configuration == ImageConfiguration::Bgra
            || image_configuration == ImageConfiguration::Rgba
        {
            let dst = &mut dst[cx * 4..];
            let src = &src[cx * 4..];
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                let rgb = Rgb::<u8>::new(
                    src[image_configuration.r_index()],
                    src[image_configuration.g_index()],
                    src[image_configuration.b_index()],
                );

                let sigmoidal = rgb.to_sigmoidal();
                dst[0] = sigmoidal.sr;
                dst[1] = sigmoidal.sg;
                dst[2] = sigmoidal.sb;
                dst[3] = src[image_configuration.a_index()] as f32 * COLOR_SCALE;
            }
        }
    }

    Ok(())
}

/// This function converts RGB to Sigmoidal. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
pub fn rgb_to_sigmoidal(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
) -> Result<(), ColorError> {
    image_to_sigmoidal::<{ ImageConfiguration::Rgb as u8 }, false>(src, dst)
}

/// This function converts BGRA to Sigmoidal. Alpha channel will be normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Sigmodal data
/// * `dst_stride` - Bytes per row for dst data
pub fn bgra_to_sigmoidal(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
) -> Result<(), ColorError> {
    image_to_sigmoidal::<{ ImageConfiguration::Bgra as u8 }, true>(src, dst)
}

/// This function converts RGBA to Sigmoidal. Alpha channel will be normalized. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive Sigmoidal data
/// * `dst_stride` - Bytes per row for dst data
pub fn rgba_to_sigmoidal(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
) -> Result<(), ColorError> {
    image_to_sigmoidal::<{ ImageConfiguration::Rgba as u8 }, true>(src, dst)
}

#[cfg(test)]
mod tests {
    use crate::BufferStore;
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_src(data: Vec<u8>, width: u32, channels: u32) -> ImageBuffer<'static, u8> {
        ImageBuffer {
            data: std::borrow::Cow::Owned(data),
            width,
            height: 1,
            channels,
            stride: width * channels,
        }
    }

    fn make_dst(width: u32, channels: u32) -> ImageBufferMut<'static, f32> {
        ImageBufferMut {
            data: BufferStore::Owned(vec![0f32; (width * channels) as usize]),
            width,
            height: 1,
            channels,
            stride: width * channels,
        }
    }

    // ── rgb_to_sigmoidal ──────────────────────────────────────────────────────

    #[test]
    fn test_rgb_to_sigmoidal_output_in_range() {
        // All output values must be in [0.0, 1.0]
        let src = make_src(vec![128, 64, 200], 1, 3);
        let mut dst = make_dst(1, 3);
        rgb_to_sigmoidal(&src, &mut dst).unwrap();
        for &v in dst.data.borrow() {
            assert!(v >= 0.0 && v <= 1.0, "Value out of range: {v}");
        }
    }

    #[test]
    fn test_rgb_to_sigmoidal_monotonic() {
        // Brighter input => higher sigmoidal output (monotonically increasing)
        let src_dark = make_src(vec![50, 50, 50], 1, 3);
        let src_mid = make_src(vec![128, 128, 128], 1, 3);
        let src_bright = make_src(vec![200, 200, 200], 1, 3);
        let mut dst_dark = make_dst(1, 3);
        let mut dst_mid = make_dst(1, 3);
        let mut dst_bright = make_dst(1, 3);
        rgb_to_sigmoidal(&src_dark, &mut dst_dark).unwrap();
        rgb_to_sigmoidal(&src_mid, &mut dst_mid).unwrap();
        rgb_to_sigmoidal(&src_bright, &mut dst_bright).unwrap();
        assert!(dst_dark.data[0] < dst_mid.data[0], "dark < mid");
        assert!(dst_mid.data[0] < dst_bright.data[0], "mid < bright");
    }

    #[test]
    fn test_rgb_to_sigmoidal_midpoint_near_half() {
        // Sigmoid midpoint: input 128 should map to ~0.5
        let src = make_src(vec![128, 128, 128], 1, 3);
        let mut dst = make_dst(1, 3);
        rgb_to_sigmoidal(&src, &mut dst).unwrap();
        for &v in dst.data.borrow() {
            assert!(
                (v - 0.5).abs() < 0.15,
                "Mid input ~128 should map near 0.5, got {v}"
            );
        }
    }

    #[test]
    fn test_rgb_to_sigmoidal_invalid_channels() {
        let src = make_src(vec![1, 2], 1, 2);
        let mut dst = make_dst(1, 2);
        let result = rgb_to_sigmoidal(&src, &mut dst);
        assert!(result.is_err(), "Should fail with 2 channels");
    }

    // ── rgba_to_sigmoidal ─────────────────────────────────────────────────────

    #[test]
    fn test_rgba_to_sigmoidal_alpha_normalized_full() {
        // Alpha 255 => 1.0
        let src = make_src(vec![128, 64, 32, 255], 1, 4);
        let mut dst = make_dst(1, 4);
        rgba_to_sigmoidal(&src, &mut dst).unwrap();
        assert!(
            (dst.data[3] - 1.0).abs() < 0.01,
            "Alpha 255 => 1.0, got {}",
            dst.data[3]
        );
    }

    #[test]
    fn test_rgba_to_sigmoidal_alpha_normalized_zero() {
        // Alpha 0 => 0.0
        let src = make_src(vec![128, 64, 32, 0], 1, 4);
        let mut dst = make_dst(1, 4);
        rgba_to_sigmoidal(&src, &mut dst).unwrap();
        assert!(
            dst.data[3].abs() < 0.01,
            "Alpha 0 => 0.0, got {}",
            dst.data[3]
        );
    }

    #[test]
    fn test_rgba_to_sigmoidal_alpha_normalized_half() {
        // Alpha 128 => ~0.502
        let src = make_src(vec![0, 0, 0, 128], 1, 4);
        let mut dst = make_dst(1, 4);
        rgba_to_sigmoidal(&src, &mut dst).unwrap();
        let expected = 128.0f32 / 255.0;
        assert!(
            (dst.data[3] - expected).abs() < 0.01,
            "Alpha 128 => ~{expected:.3}, got {}",
            dst.data[3]
        );
    }

    #[test]
    fn test_rgba_to_sigmoidal_rgb_channels_unaffected_by_alpha() {
        // RGB sigmoidal values must be the same regardless of alpha
        let src_a255 = make_src(vec![200, 100, 50, 255], 1, 4);
        let src_a0 = make_src(vec![200, 100, 50, 0], 1, 4);
        let mut dst_a255 = make_dst(1, 4);
        let mut dst_a0 = make_dst(1, 4);
        rgba_to_sigmoidal(&src_a255, &mut dst_a255).unwrap();
        rgba_to_sigmoidal(&src_a0, &mut dst_a0).unwrap();
        for i in 0..3 {
            assert!(
                (dst_a255.data[i] - dst_a0.data[i]).abs() < 1e-6,
                "Channel {i} should not depend on alpha"
            );
        }
    }

    #[test]
    fn test_rgba_to_sigmoidal_output_in_range() {
        let src = make_src(vec![10, 200, 75, 180], 1, 4);
        let mut dst = make_dst(1, 4);
        rgba_to_sigmoidal(&src, &mut dst).unwrap();
        for &v in dst.data.borrow() {
            assert!(v >= 0.0 && v <= 1.0, "Value out of range: {v}");
        }
    }

    #[test]
    fn test_bgra_to_sigmoidal_channel_swap() {
        // Pure red in RGBA = (255,0,0,255); pure red in BGRA = (0,0,255,255)
        // Both represent the same color — sr should match
        let src_rgba = make_src(vec![255, 0, 0, 255], 1, 4);
        let src_bgra = make_src(vec![0, 0, 255, 255], 1, 4);
        let mut dst_rgba = make_dst(1, 4);
        let mut dst_bgra = make_dst(1, 4);
        rgba_to_sigmoidal(&src_rgba, &mut dst_rgba).unwrap();
        bgra_to_sigmoidal(&src_bgra, &mut dst_bgra).unwrap();
        assert!(
            (dst_rgba.data[0] - dst_bgra.data[0]).abs() < 1e-5,
            "sr mismatch: rgba={} bgra={}",
            dst_rgba.data[0],
            dst_bgra.data[0]
        );
        assert!(
            (dst_rgba.data[1] - dst_bgra.data[1]).abs() < 1e-5,
            "sg mismatch"
        );
        assert!(
            (dst_rgba.data[2] - dst_bgra.data[2]).abs() < 1e-5,
            "sb mismatch"
        );
    }

    #[test]
    fn test_bgra_to_sigmoidal_alpha_normalized() {
        let src = make_src(vec![0, 0, 0, 200], 1, 4);
        let mut dst = make_dst(1, 4);
        bgra_to_sigmoidal(&src, &mut dst).unwrap();
        let expected = 200.0f32 / 255.0;
        assert!(
            (dst.data[3] - expected).abs() < 0.01,
            "Alpha ~{expected:.3}, got {}",
            dst.data[3]
        );
    }

    #[test]
    fn test_bgra_to_sigmoidal_neutral_matches_rgba() {
        // For neutral grey BGR==RGB so both should give identical results
        let src_rgba = make_src(vec![128, 128, 128, 128], 1, 4);
        let src_bgra = make_src(vec![128, 128, 128, 128], 1, 4);
        let mut dst_rgba = make_dst(1, 4);
        let mut dst_bgra = make_dst(1, 4);
        rgba_to_sigmoidal(&src_rgba, &mut dst_rgba).unwrap();
        bgra_to_sigmoidal(&src_bgra, &mut dst_bgra).unwrap();
        for i in 0..4 {
            assert!(
                (dst_rgba.data[i] - dst_bgra.data[i]).abs() < 1e-5,
                "Channel {i} mismatch for neutral grey"
            );
        }
    }

    #[test]
    fn test_bgra_to_sigmoidal_output_in_range() {
        let src = make_src(vec![33, 177, 210, 99], 1, 4);
        let mut dst = make_dst(1, 4);
        bgra_to_sigmoidal(&src, &mut dst).unwrap();
        for &v in dst.data.borrow() {
            assert!(v >= 0.0 && v <= 1.0, "Value out of range: {v}");
        }
    }

    #[test]
    fn test_rgb_and_rgba_same_rgb_channels() {
        // RGB channels must be identical whether or not alpha is present
        let src_rgb = make_src(vec![100, 150, 200], 1, 3);
        let src_rgba = make_src(vec![100, 150, 200, 255], 1, 4);
        let mut dst_rgb = make_dst(1, 3);
        let mut dst_rgba = make_dst(1, 4);
        rgb_to_sigmoidal(&src_rgb, &mut dst_rgb).unwrap();
        rgba_to_sigmoidal(&src_rgba, &mut dst_rgba).unwrap();
        for i in 0..3 {
            assert!(
                (dst_rgb.data[i] - dst_rgba.data[i]).abs() < 1e-5,
                "Channel {i} differs between RGB and RGBA"
            );
        }
    }
}
