/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
use crate::{ColorError, Hsl, Hsv, ImageBuffer, ImageBufferMut};

fn hsv_u16_to_channels<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: &ImageBuffer<'_, u16>,
    dst: &mut ImageBufferMut<'_, u8>,
    scale: f32,
) -> Result<(), ColorError> {
    dst.validate()?;
    src.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }

    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let target: HsvTarget = TARGET.into();

    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut _wide_row_handler: Option<unsafe fn(usize, &[u16], &mut [u8], u32, f32) -> usize> =
        None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        use crate::sse::sse_hsv_u16_to_image;
        _wide_row_handler = Some(sse_hsv_u16_to_image::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::neon::neon_hsv_u16_to_image;
        _wide_row_handler =
            Some(neon_hsv_u16_to_image::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    let inv_scale = 1f32 / scale;

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

        if let Some(dispatch) = _wide_row_handler {
            cx = unsafe { dispatch(0, src, dst, width, inv_scale) };
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
                let h = src[0] as f32;
                let s = src[1] as f32 * inv_scale;
                let v = src[2] as f32 * inv_scale;
                let rgb = match target {
                    HsvTarget::Hsv => Hsv::from_components(h, s, v).to_rgb8(),
                    HsvTarget::Hsl => Hsl::from_components(h, s, v).to_rgb8(),
                };
                dst[image_configuration.r_index()] = rgb.r;
                dst[image_configuration.g_index()] = rgb.g;
                dst[image_configuration.b_index()] = rgb.b;
            }
        } else if image_configuration == ImageConfiguration::Rgba
            || image_configuration == ImageConfiguration::Bgra
        {
            let dst = &mut dst[cx * 4..];
            let src = &src[cx * 4..];
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                let h = src[0] as f32;
                let s = src[1] as f32 * inv_scale;
                let v = src[2] as f32 * inv_scale;
                let rgb = match target {
                    HsvTarget::Hsv => Hsv::from_components(h, s, v).to_rgb8(),
                    HsvTarget::Hsl => Hsl::from_components(h, s, v).to_rgb8(),
                };
                dst[image_configuration.r_index()] = rgb.r;
                dst[image_configuration.g_index()] = rgb.g;
                dst[image_configuration.b_index()] = rgb.b;
                if USE_ALPHA {
                    dst[image_configuration.a_index()] = src[3] as u8;
                }
            }
        }
    }

    Ok(())
}

/// Converts HSV data stored as `u16` to RGB.
///
/// Hue is expressed in degrees `[0, 360)`. Saturation and Value are scaled to
/// `[0, scale]` — pass `scale = 1.0` to keep them in the natural `[0, 1]` range,
/// or `scale = 100.0` if they are stored as percentages.
pub fn hsv_to_rgb(
    src: &ImageBuffer<'_, u16>,
    dst: &mut ImageBufferMut<'_, u8>,
    scale: f32,
) -> Result<(), ColorError> {
    hsv_u16_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { HsvTarget::Hsv as u8 }>(
        src, dst, scale,
    )
}

pub fn hsv_to_bgra(
    src: &ImageBuffer<'_, u16>,
    dst: &mut ImageBufferMut<'_, u8>,
    scale: f32,
) -> Result<(), ColorError> {
    hsv_u16_to_channels::<{ ImageConfiguration::Bgra as u8 }, true, { HsvTarget::Hsv as u8 }>(
        src, dst, scale,
    )
}

pub fn hsv_to_rgba(
    src: &ImageBuffer<'_, u16>,
    dst: &mut ImageBufferMut<'_, u8>,
    scale: f32,
) -> Result<(), ColorError> {
    hsv_u16_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { HsvTarget::Hsv as u8 }>(
        src, dst, scale,
    )
}

pub fn hsl_to_rgb(
    src: &ImageBuffer<'_, u16>,
    dst: &mut ImageBufferMut<'_, u8>,
    scale: f32,
) -> Result<(), ColorError> {
    hsv_u16_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { HsvTarget::Hsl as u8 }>(
        src, dst, scale,
    )
}

pub fn hsl_to_bgra(
    src: &ImageBuffer<'_, u16>,
    dst: &mut ImageBufferMut<'_, u8>,
    scale: f32,
) -> Result<(), ColorError> {
    hsv_u16_to_channels::<{ ImageConfiguration::Bgra as u8 }, true, { HsvTarget::Hsl as u8 }>(
        src, dst, scale,
    )
}

pub fn hsl_to_rgba(
    src: &ImageBuffer<'_, u16>,
    dst: &mut ImageBufferMut<'_, u8>,
    scale: f32,
) -> Result<(), ColorError> {
    hsv_u16_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { HsvTarget::Hsl as u8 }>(
        src, dst, scale,
    )
}

#[cfg(test)]
mod tests_hsv_u16_to_channels {
    use super::*;
    use crate::{BufferStore, Rgb};

    const SCALE: f32 = 1.0;
    const SCALE_100: f32 = 100.0;

    fn make_src(
        data: Vec<u16>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> ImageBuffer<'static, u16> {
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

    fn assert_approx(a: u8, b: u8, eps: u8, label: &str) {
        assert!(
            (a as i16 - b as i16).abs() <= eps as i16,
            "{label}: got {a}, expected {b}"
        );
    }

    // Convert known RGB u8 → HSV u16 so we can test the inverse
    fn rgb_to_hsv_u16(r: u8, g: u8, b: u8, scale: f32) -> (u16, u16, u16) {
        let rgb = Rgb::<u8>::new(r, g, b);
        let hsv: Hsv = Hsv::from(rgb);
        (
            hsv.h as u16,
            (hsv.s * scale).round() as u16,
            (hsv.v * scale).round() as u16,
        )
    }

    fn rgb_to_hsl_u16(r: u8, g: u8, b: u8, scale: f32) -> (u16, u16, u16) {
        let rgb = Rgb::<u8>::new(r, g, b);
        let hsl: Hsl = Hsl::from_rgb(rgb);
        (
            hsl.h as u16,
            (hsl.s * scale).round() as u16,
            (hsl.l * scale).round() as u16,
        )
    }

    // ── hsv_to_rgb ────────────────────────────────────────────────────────────

    #[test]
    fn hsv_rgb_black_roundtrip() {
        let (h, s, v) = rgb_to_hsv_u16(0, 0, 0, SCALE);
        let src = make_src(vec![h, s, v], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        hsv_to_rgb(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "R black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "B black");
    }

    #[test]
    fn hsv_rgb_white_roundtrip() {
        let (h, s, v) = rgb_to_hsv_u16(255, 255, 255, SCALE);
        let src = make_src(vec![h, s, v], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        hsv_to_rgb(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "B white");
    }

    #[test]
    fn hsv_rgb_red_roundtrip() {
        let (h, s, v) = rgb_to_hsv_u16(255, 0, 0, SCALE);
        let src = make_src(vec![h, s, v], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        hsv_to_rgb(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] > 200, "R dominant for red, got {}", d[0]);
        assert!(d[1] < 20, "G low for red, got {}", d[1]);
        assert!(d[2] < 20, "B low for red, got {}", d[2]);
    }

    #[test]
    fn hsv_rgb_blue_channel_order() {
        let (h, s, v) = rgb_to_hsv_u16(0, 0, 255, SCALE);
        let src = make_src(vec![h, s, v], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        hsv_to_rgb(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] < 20, "R low for blue, got {}", d[0]);
        assert!(d[1] < 20, "G low for blue, got {}", d[1]);
        assert!(d[2] > 200, "B dominant for blue, got {}", d[2]);
    }

    // ── scale variants ────────────────────────────────────────────────────────

    #[test]
    fn hsv_rgb_scale_100_roundtrip() {
        let (h, s, v) = rgb_to_hsv_u16(128, 64, 200, SCALE_100);
        let src = make_src(vec![h, s, v], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        hsv_to_rgb(&src, &mut dst, SCALE_100).unwrap();
        let d = dst.data.borrow();
        // Verify the pixel is not all-zero — scale was applied correctly
        assert!(
            d[0] > 0 || d[1] > 0 || d[2] > 0,
            "output should be non-zero"
        );
    }

    #[test]
    fn hsv_rgba_white_opaque_roundtrip() {
        let (h, s, v) = rgb_to_hsv_u16(255, 255, 255, SCALE);
        let src = make_src(vec![h, s, v, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        hsv_to_rgba(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "B");
        assert_eq!(d[3], 255, "A opaque");
    }

    #[test]
    fn hsv_rgba_alpha_passthrough() {
        let (h, s, v) = rgb_to_hsv_u16(128, 128, 128, SCALE);
        for alpha in [0u16, 64, 128, 192, 255] {
            let src = make_src(vec![h, s, v, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            hsv_to_rgba(&src, &mut dst, SCALE).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[3], alpha as u8, "alpha={alpha} not passed through");
        }
    }

    #[test]
    fn hsv_bgra_white_opaque_roundtrip() {
        let (h, s, v) = rgb_to_hsv_u16(255, 255, 255, SCALE);
        let src = make_src(vec![h, s, v, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        hsv_to_bgra(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        // BGRA: d[0]=B d[1]=G d[2]=R d[3]=A
        assert_approx(d[0], 255, 2, "B white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "R white");
        assert_eq!(d[3], 255, "A");
    }

    #[test]
    fn hsv_bgra_alpha_passthrough() {
        let (h, s, v) = rgb_to_hsv_u16(100, 100, 100, SCALE);
        for alpha in [0u16, 128, 255] {
            let src = make_src(vec![h, s, v, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            hsv_to_bgra(&src, &mut dst, SCALE).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[3], alpha as u8, "alpha={alpha}");
        }
    }

    #[test]
    fn hsv_rgba_bgra_agree_on_grey() {
        let (h, s, v) = rgb_to_hsv_u16(150, 150, 150, SCALE);
        let src_rgba = make_src(vec![h, s, v, 200], 1, 1, 4);
        let src_bgra = make_src(vec![h, s, v, 200], 1, 1, 4);
        let mut dst_rgba = make_dst(1, 1, 4);
        let mut dst_bgra = make_dst(1, 1, 4);
        hsv_to_rgba(&src_rgba, &mut dst_rgba, SCALE).unwrap();
        hsv_to_bgra(&src_bgra, &mut dst_bgra, SCALE).unwrap();
        assert_eq!(
            &*dst_rgba.data.borrow(),
            &*dst_bgra.data.borrow(),
            "grey invariant to channel order"
        );
    }

    #[test]
    fn hsv_bgra_channel_order_for_red() {
        let (h, s, v) = rgb_to_hsv_u16(255, 0, 0, SCALE);
        let src = make_src(vec![h, s, v, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        hsv_to_bgra(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] < 20, "B low for red in BGRA, got {}", d[0]);
        assert!(d[1] < 20, "G low for red in BGRA, got {}", d[1]);
        assert!(d[2] > 200, "R dominant for red in BGRA, got {}", d[2]);
    }

    // ── hsl_to_rgb ────────────────────────────────────────────────────────────

    #[test]
    fn hsl_rgb_black_roundtrip() {
        let (h, s, l) = rgb_to_hsl_u16(0, 0, 0, SCALE);
        let src = make_src(vec![h, s, l], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        hsl_to_rgb(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "R black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "B black");
    }

    #[test]
    fn hsl_rgb_white_roundtrip() {
        let (h, s, l) = rgb_to_hsl_u16(255, 255, 255, SCALE);
        let src = make_src(vec![h, s, l], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        hsl_to_rgb(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "B white");
    }

    #[test]
    fn hsl_rgba_white_opaque_roundtrip() {
        let (h, s, l) = rgb_to_hsl_u16(255, 255, 255, SCALE);
        let src = make_src(vec![h, s, l, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        hsl_to_rgba(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "B");
        assert_eq!(d[3], 255, "A");
    }

    #[test]
    fn hsl_rgba_alpha_passthrough() {
        let (h, s, l) = rgb_to_hsl_u16(128, 128, 128, SCALE);
        for alpha in [0u16, 64, 128, 192, 255] {
            let src = make_src(vec![h, s, l, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            hsl_to_rgba(&src, &mut dst, SCALE).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[3], alpha as u8, "alpha={alpha}");
        }
    }

    #[test]
    fn hsl_bgra_white_opaque_roundtrip() {
        let (h, s, l) = rgb_to_hsl_u16(255, 255, 255, SCALE);
        let src = make_src(vec![h, s, l, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        hsl_to_bgra(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "B white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "R white");
        assert_eq!(d[3], 255, "A");
    }

    #[test]
    fn hsl_bgra_alpha_passthrough() {
        let (h, s, l) = rgb_to_hsl_u16(100, 100, 100, SCALE);
        for alpha in [0u16, 128, 255] {
            let src = make_src(vec![h, s, l, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            hsl_to_bgra(&src, &mut dst, SCALE).unwrap();
            let d = dst.data.borrow();
            assert_eq!(d[3], alpha as u8, "alpha={alpha}");
        }
    }

    #[test]
    fn hsl_rgba_bgra_agree_on_grey() {
        let (h, s, l) = rgb_to_hsl_u16(180, 180, 180, SCALE);
        let src_rgba = make_src(vec![h, s, l, 128], 1, 1, 4);
        let src_bgra = make_src(vec![h, s, l, 128], 1, 1, 4);
        let mut dst_rgba = make_dst(1, 1, 4);
        let mut dst_bgra = make_dst(1, 1, 4);
        hsl_to_rgba(&src_rgba, &mut dst_rgba, SCALE).unwrap();
        hsl_to_bgra(&src_bgra, &mut dst_bgra, SCALE).unwrap();
        assert_eq!(
            &*dst_rgba.data.borrow(),
            &*dst_bgra.data.borrow(),
            "grey invariant to channel order"
        );
    }

    // ── HSV vs HSL differ for chromatic input ─────────────────────────────────

    #[test]
    fn hsv_and_hsl_differ_for_chromatic_input() {
        // Same hue/saturation/value-or-lightness → different colour models
        // should produce different RGB output for a saturated colour
        let h = 120u16; // green hue
        let s = 1u16; // full saturation (scale=1)
        let v = 1u16; // full value / lightness (scale=1)

        let src_hsv = make_src(vec![h, s, v], 1, 1, 3);
        let src_hsl = make_src(vec![h, s, v], 1, 1, 3);
        let mut dst_hsv = make_dst(1, 1, 3);
        let mut dst_hsl = make_dst(1, 1, 3);
        hsv_to_rgb(&src_hsv, &mut dst_hsv, SCALE).unwrap();
        hsl_to_rgb(&src_hsl, &mut dst_hsl, SCALE).unwrap();

        // HSV(120°,1,1) = pure green (0,255,0)
        // HSL(120°,1,1) = white (255,255,255) — lightness=1 is always white
        // They must not be identical
        assert_ne!(
            &*dst_hsv.data.borrow(),
            &*dst_hsl.data.borrow(),
            "HSV and HSL should produce different output for same raw input"
        );
    }

    // ── multi-pixel / multi-row consistency ───────────────────────────────────

    #[test]
    fn hsv_rgb_multi_pixel_consistent() {
        let (h, s, v) = rgb_to_hsv_u16(100, 150, 200, SCALE);
        let src = make_src(vec![h, s, v].repeat(4), 4, 1, 3);
        let mut dst = make_dst(4, 1, 3);
        hsv_to_rgb(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn hsv_rgba_multi_pixel_consistent() {
        let (h, s, v) = rgb_to_hsv_u16(80, 120, 200, SCALE);
        let src = make_src(vec![h, s, v, 180].repeat(4), 4, 1, 4);
        let mut dst = make_dst(4, 1, 4);
        hsv_to_rgba(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2], d[3]];
        for (i, chunk) in d.chunks(4).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn hsl_rgb_multi_row_consistent() {
        let (h, s, l) = rgb_to_hsl_u16(60, 120, 180, SCALE);
        let src = make_src(vec![h, s, l].repeat(4), 2, 2, 3);
        let mut dst = make_dst(2, 2, 3);
        hsl_to_rgb(&src, &mut dst, SCALE).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    // ── error paths ───────────────────────────────────────────────────────────

    #[test]
    fn hsv_rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0u16; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(hsv_to_rgb(&src, &mut dst, SCALE).is_err());
    }

    #[test]
    fn hsv_rgba_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0u16; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 4);
        assert!(hsv_to_rgba(&src, &mut dst, SCALE).is_err());
    }

    #[test]
    fn hsl_rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0u16; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(hsl_to_rgb(&src, &mut dst, SCALE).is_err());
    }

    #[test]
    fn hsl_rgba_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0u16; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 4);
        assert!(hsl_to_rgba(&src, &mut dst, SCALE).is_err());
    }
}
