/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::{ColorError, ImageBuffer, ImageBufferMut, Rgb, Sigmoidal};

fn sigmoidal_to_image<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool>(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
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

    let mut _wide_row_handler: Option<unsafe fn(usize, &[f32], &mut [u8], u32) -> usize> = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        use crate::sse::sse_from_sigmoidal_row;
        _wide_row_handler = Some(sse_from_sigmoidal_row::<CHANNELS_CONFIGURATION>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        use crate::avx::avx_from_sigmoidal_row;
        _wide_row_handler = Some(avx_from_sigmoidal_row::<CHANNELS_CONFIGURATION>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::neon::neon_from_sigmoidal_row;
        _wide_row_handler = Some(neon_from_sigmoidal_row::<CHANNELS_CONFIGURATION>);
    }

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
            let src = &src[cx * 3..];
            let dst = &mut dst[cx * 3..];
            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                let sigmoidal = Sigmoidal::new(src[0], src[1], src[2]);
                let rgb: Rgb<u8> = sigmoidal.into();
                dst[image_configuration.r_index()] = rgb.r;
                dst[image_configuration.g_index()] = rgb.g;
                dst[image_configuration.b_index()] = rgb.b;
            }
        } else if image_configuration == ImageConfiguration::Rgba
            || image_configuration == ImageConfiguration::Bgra
        {
            let src = &src[cx * 4..];
            let dst = &mut dst[cx * 4..];
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                let sigmoidal = Sigmoidal::new(src[0], src[1], src[2]);
                let rgb: Rgb<u8> = sigmoidal.into();
                dst[image_configuration.r_index()] = rgb.r;
                dst[image_configuration.g_index()] = rgb.g;
                dst[image_configuration.b_index()] = rgb.b;
                dst[image_configuration.a_index()] =
                    (src[3] * 255f32).clamp(0f32, 255f32).round() as u8;
            }
        }
    }

    Ok(())
}

/// This function converts Sigmoid to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
) -> Result<(), ColorError> {
    sigmoidal_to_image::<{ ImageConfiguration::Rgb as u8 }, false>(src, dst)
}

/// This function converts Sigmoid to BGRA. Alpha channel expected to be normalized and will be denormalized during transformation. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
) -> Result<(), ColorError> {
    sigmoidal_to_image::<{ ImageConfiguration::Bgra as u8 }, true>(src, dst)
}

/// This function converts Sigmoid to RGBA. Alpha channel expected to be normalized and will be denormalized during transformation. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains HSV data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
pub fn sigmoidal_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
) -> Result<(), ColorError> {
    sigmoidal_to_image::<{ ImageConfiguration::Rgba as u8 }, true>(src, dst)
}

#[cfg(test)]
mod tests_sigmoidal_to_image {
    use super::*;
    use crate::BufferStore;

    fn make_src(
        data: Vec<f32>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> ImageBuffer<'static, f32> {
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

    fn assert_approx(a: u8, b: u8, eps: u8, label: &str) {
        assert!(
            (a as i16 - b as i16).abs() <= eps as i16,
            "{label}: got {a}, expected {b}"
        );
    }

    // Convert known RGB u8 → Sigmoidal f32 so tests can work with the inverse
    fn rgb_to_sigmoidal(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
        let rgb = Rgb::<u8>::new(r, g, b);
        let sig: Sigmoidal = rgb.into();
        (sig.sr, sig.sg, sig.sb)
    }

    // ── black / white roundtrip ───────────────────────────────────────────────

    #[test]
    fn rgb_black_roundtrip() {
        let (sr, sg, sb) = rgb_to_sigmoidal(0, 0, 0);
        let src = make_src(vec![sr, sg, sb], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        sigmoidal_to_rgb(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 1, "R black");
        assert_approx(d[1], 0, 1, "G black");
        assert_approx(d[2], 0, 1, "B black");
    }

    #[test]
    fn rgb_white_roundtrip() {
        let (sr, sg, sb) = rgb_to_sigmoidal(255, 255, 255);
        let src = make_src(vec![sr, sg, sb], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        sigmoidal_to_rgb(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 1, "R white");
        assert_approx(d[1], 255, 1, "G white");
        assert_approx(d[2], 255, 1, "B white");
    }

    #[test]
    fn rgba_black_opaque_roundtrip() {
        let (sr, sg, sb) = rgb_to_sigmoidal(0, 0, 0);
        let src = make_src(vec![sr, sg, sb, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        sigmoidal_to_rgba(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 1, "R");
        assert_approx(d[1], 0, 1, "G");
        assert_approx(d[2], 0, 1, "B");
        assert_approx(d[3], 255, 1, "A");
    }

    #[test]
    fn rgba_white_opaque_roundtrip() {
        let (sr, sg, sb) = rgb_to_sigmoidal(255, 255, 255);
        let src = make_src(vec![sr, sg, sb, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        sigmoidal_to_rgba(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 1, "R");
        assert_approx(d[1], 255, 1, "G");
        assert_approx(d[2], 255, 1, "B");
        assert_approx(d[3], 255, 1, "A");
    }

    #[test]
    fn bgra_white_opaque_roundtrip() {
        let (sr, sg, sb) = rgb_to_sigmoidal(255, 255, 255);
        let src = make_src(vec![sr, sg, sb, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        sigmoidal_to_bgra(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        // BGRA layout: d[0]=B d[1]=G d[2]=R d[3]=A
        assert_approx(d[0], 255, 1, "B");
        assert_approx(d[1], 255, 1, "G");
        assert_approx(d[2], 255, 1, "R");
        assert_approx(d[3], 255, 1, "A");
    }

    // ── full roundtrip for a range of values ──────────────────────────────────

    #[test]
    fn rgb_roundtrip_range() {
        for v in (0u8..=255).step_by(16) {
            let (sr, sg, sb) = rgb_to_sigmoidal(v, v, v);
            let src = make_src(vec![sr, sg, sb], 1, 1, 3);
            let mut dst = make_dst(1, 1, 3);
            sigmoidal_to_rgb(&src, &mut dst).unwrap();
            let d = dst.data.borrow();
            assert_approx(d[0], v, 1, &format!("R v={v}"));
            assert_approx(d[1], v, 1, &format!("G v={v}"));
            assert_approx(d[2], v, 1, &format!("B v={v}"));
        }
    }

    #[test]
    fn rgba_roundtrip_range() {
        for v in (0u8..=255).step_by(32) {
            let (sr, sg, sb) = rgb_to_sigmoidal(v, v, v);
            let src = make_src(vec![sr, sg, sb, 0.6], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            sigmoidal_to_rgba(&src, &mut dst).unwrap();
            let d = dst.data.borrow();
            assert_approx(d[0], v, 1, &format!("R v={v}"));
            assert_approx(d[1], v, 1, &format!("G v={v}"));
            assert_approx(d[2], v, 1, &format!("B v={v}"));
        }
    }

    // ── alpha denormalization ─────────────────────────────────────────────────

    #[test]
    fn rgba_alpha_denormalized_correctly() {
        let (sr, sg, sb) = rgb_to_sigmoidal(128, 128, 128);
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src(vec![sr, sg, sb, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            sigmoidal_to_rgba(&src, &mut dst).unwrap();
            let d = dst.data.borrow();
            let expected = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected, 1, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn bgra_alpha_denormalized_correctly() {
        let (sr, sg, sb) = rgb_to_sigmoidal(128, 128, 128);
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src(vec![sr, sg, sb, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            sigmoidal_to_bgra(&src, &mut dst).unwrap();
            let d = dst.data.borrow();
            let expected = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected, 1, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn rgba_rgb_channels_encoded_when_alpha_zero() {
        let (sr, sg, sb) = rgb_to_sigmoidal(200, 100, 50);
        let src = make_src(vec![sr, sg, sb, 0.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        sigmoidal_to_rgba(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 200, 1, "R encoded despite alpha=0");
        assert_eq!(d[3], 0, "A must remain 0");
    }

    // ── channel ordering ──────────────────────────────────────────────────────

    #[test]
    fn rgb_channel_order_correct_for_red() {
        let (sr, sg, sb) = rgb_to_sigmoidal(255, 0, 0);
        let src = make_src(vec![sr, sg, sb], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        sigmoidal_to_rgb(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] > 200, "R dominant for red input, got {}", d[0]);
        assert!(d[1] < 10, "G low for red input, got {}", d[1]);
        assert!(d[2] < 10, "B low for red input, got {}", d[2]);
    }

    #[test]
    fn bgra_channel_order_correct_for_red() {
        // BGRA layout: d[0]=B d[1]=G d[2]=R d[3]=A
        let (sr, sg, sb) = rgb_to_sigmoidal(255, 0, 0);
        let src = make_src(vec![sr, sg, sb, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        sigmoidal_to_bgra(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] < 10, "B low for red input in BGRA, got {}", d[0]);
        assert!(d[1] < 10, "G low for red input in BGRA, got {}", d[1]);
        assert!(d[2] > 200, "R dominant for red input in BGRA, got {}", d[2]);
        assert_approx(d[3], 255, 1, "A");
    }

    #[test]
    fn rgba_channel_order_correct_for_blue() {
        let (sr, sg, sb) = rgb_to_sigmoidal(0, 0, 255);
        let src = make_src(vec![sr, sg, sb, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        sigmoidal_to_rgba(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] < 10, "R low for blue input, got {}", d[0]);
        assert!(d[1] < 10, "G low for blue input, got {}", d[1]);
        assert!(d[2] > 200, "B dominant for blue input, got {}", d[2]);
    }

    // ── rgb and bgra agree on grey ────────────────────────────────────────────

    #[test]
    fn rgb_rgba_agree_on_grey_channels() {
        let (sr, sg, sb) = rgb_to_sigmoidal(128, 128, 128);
        let src_rgb = make_src(vec![sr, sg, sb], 1, 1, 3);
        let src_rgba = make_src(vec![sr, sg, sb, 1.0], 1, 1, 4);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_rgba = make_dst(1, 1, 4);
        sigmoidal_to_rgb(&src_rgb, &mut dst_rgb).unwrap();
        sigmoidal_to_rgba(&src_rgba, &mut dst_rgba).unwrap();
        let r = dst_rgb.data.borrow();
        let a = dst_rgba.data.borrow();
        assert_approx(r[0], a[0], 1, "R");
        assert_approx(r[1], a[1], 1, "G");
        assert_approx(r[2], a[2], 1, "B");
    }

    #[test]
    fn rgba_bgra_agree_on_grey() {
        let (sr, sg, sb) = rgb_to_sigmoidal(180, 180, 180);
        let src_rgba = make_src(vec![sr, sg, sb, 0.5], 1, 1, 4);
        let src_bgra = make_src(vec![sr, sg, sb, 0.5], 1, 1, 4);
        let mut dst_rgba = make_dst(1, 1, 4);
        let mut dst_bgra = make_dst(1, 1, 4);
        sigmoidal_to_rgba(&src_rgba, &mut dst_rgba).unwrap();
        sigmoidal_to_bgra(&src_bgra, &mut dst_bgra).unwrap();
        assert_eq!(
            &*dst_rgba.data.borrow(),
            &*dst_bgra.data.borrow(),
            "grey must be invariant to channel order"
        );
    }

    // ── multi-pixel / multi-row consistency ───────────────────────────────────

    #[test]
    fn rgb_multi_pixel_consistent() {
        let (sr, sg, sb) = rgb_to_sigmoidal(100, 150, 200);
        let src = make_src(vec![sr, sg, sb].repeat(4), 4, 1, 3);
        let mut dst = make_dst(4, 1, 3);
        sigmoidal_to_rgb(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn rgba_multi_pixel_consistent() {
        let (sr, sg, sb) = rgb_to_sigmoidal(80, 120, 200);
        let src = make_src(vec![sr, sg, sb, 0.7].repeat(4), 4, 1, 4);
        let mut dst = make_dst(4, 1, 4);
        sigmoidal_to_rgba(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2], d[3]];
        for (i, chunk) in d.chunks(4).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn rgb_multi_row_consistent() {
        let (sr, sg, sb) = rgb_to_sigmoidal(60, 120, 180);
        let src = make_src(vec![sr, sg, sb].repeat(4), 2, 2, 3);
        let mut dst = make_dst(2, 2, 3);
        sigmoidal_to_rgb(&src, &mut dst).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(sigmoidal_to_rgb(&src, &mut dst).is_err());
    }

    #[test]
    fn rgba_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2).unwrap();
        let mut dst = make_dst(1, 1, 4);
        assert!(sigmoidal_to_rgba(&src, &mut dst).is_err());
    }

    #[test]
    fn bgra_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2).unwrap();
        let mut dst = make_dst(1, 1, 4);
        assert!(sigmoidal_to_bgra(&src, &mut dst).is_err());
    }
}
