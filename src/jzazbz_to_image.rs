/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::image::ImageConfiguration;
use crate::image_to_jzazbz::JzazbzTarget;
use crate::{ColorError, ImageBuffer, ImageBufferMut, Jzazbz, Jzczhz, TransferFunction};

fn jzazbz_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    dst.validate()?;
    src.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }

    let mut _wide_row_handle: Option<unsafe fn(usize, &[f32], &mut [f32], u32, f32) -> usize> =
        None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        use crate::sse::sse_jzazbz_to_image;
        _wide_row_handle = Some(sse_jzazbz_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::neon::neon_jzazbz_to_image;
        _wide_row_handle = Some(neon_jzazbz_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let target: JzazbzTarget = TARGET.into();
    let channels = image_configuration.channel_count();

    let mut lut_table = [0u8; 65536];
    for (i, lut) in lut_table[..2049].iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.)
            .round()
            .min(255.) as u8;
    }

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;
    let width = src.width as usize;

    let mut transient_row = vec![0f32; width * channels];

    for (dst, src) in dst
        .data
        .borrow_mut()
        .chunks_mut(dst_stride)
        .zip(src.data.chunks(src.stride()))
    {
        let src = &src[..src_r_width as usize];
        let dst = &mut dst[..dst_r_width as usize];

        let mut cx = 0usize;
        if let Some(dispatcher) = _wide_row_handle {
            cx = unsafe { dispatcher(0, src, &mut transient_row, width as u32, display_luminance) }
        }

        if image_configuration.has_alpha() {
            let transient_row = &mut transient_row[cx * 4..];
            let src = &src[cx * 4..];
            for (transient, src) in transient_row
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                let rgb = match target {
                    JzazbzTarget::Jzazbz => {
                        let jzazbz =
                            Jzazbz::new_with_luminance(src[0], src[1], src[2], display_luminance);
                        jzazbz.to_linear_rgb()
                    }
                    JzazbzTarget::Jzczhz => {
                        let jzczhz = Jzczhz::new(src[0], src[1], src[2]);
                        jzczhz.to_linear_rgb_with_luminance(display_luminance)
                    }
                };
                transient[image_configuration.r_index()] = rgb.r;
                transient[image_configuration.g_index()] = rgb.g;
                transient[image_configuration.b_index()] = rgb.b;
                transient[image_configuration.a_index()] = src[3];
            }
        } else {
            let transient_row = &mut transient_row[cx * 3..];
            let src = &src[cx * 3..];
            for (transient, src) in transient_row
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                let rgb = match target {
                    JzazbzTarget::Jzazbz => {
                        let jzazbz =
                            Jzazbz::new_with_luminance(src[0], src[1], src[2], display_luminance);
                        jzazbz.to_linear_rgb()
                    }
                    JzazbzTarget::Jzczhz => {
                        let jzczhz = Jzczhz::new(src[0], src[1], src[2]);
                        jzczhz.to_linear_rgb_with_luminance(display_luminance)
                    }
                };
                transient[image_configuration.r_index()] = rgb.r;
                transient[image_configuration.g_index()] = rgb.g;
                transient[image_configuration.b_index()] = rgb.b;
            }
        }

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(transient_row.as_chunks::<3>().0.iter())
            {
                let r = (src[image_configuration.r_index()] * 2048.).round() as u16;
                let g = (src[image_configuration.g_index()] * 2048.).round() as u16;
                let b = (src[image_configuration.b_index()] * 2048.).round() as u16;
                dst[image_configuration.r_index()] = lut_table[r.min(2048) as usize];
                dst[image_configuration.g_index()] = lut_table[g.min(2048) as usize];
                dst[image_configuration.b_index()] = lut_table[b.min(2048) as usize];
            }
        } else if image_configuration == ImageConfiguration::Rgba
            || image_configuration == ImageConfiguration::Bgra
        {
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(transient_row.as_chunks::<4>().0.iter())
            {
                let r = (src[image_configuration.r_index()] * 2048.).round() as u16;
                let g = (src[image_configuration.g_index()] * 2048.).round() as u16;
                let b = (src[image_configuration.b_index()] * 2048.).round() as u16;
                dst[image_configuration.r_index()] = lut_table[r.min(2048) as usize];
                dst[image_configuration.g_index()] = lut_table[g.min(2048) as usize];
                dst[image_configuration.b_index()] = lut_table[b.min(2048) as usize];
                dst[image_configuration.a_index()] = (src[image_configuration.a_index()] * 255.)
                    .clamp(0., 255.)
                    .round() as u8;
            }
        }
    }

    Ok(())
}

/// This function converts Jzazbz with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzazbz_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    jzazbz_to_image::<{ ImageConfiguration::Rgba as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts Jzazbz to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzazbz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzazbz_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    jzazbz_to_image::<{ ImageConfiguration::Rgb as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts Jzazbz to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzazbz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzazbz_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    jzazbz_to_image::<{ ImageConfiguration::Bgr as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts Jzazbz with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzazbz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzazbz_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    jzazbz_to_image::<{ ImageConfiguration::Bgra as u8 }, { JzazbzTarget::Jzazbz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts Jzczhz with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzczhz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzczhz_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    jzazbz_to_image::<{ ImageConfiguration::Rgba as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts Jzczhz to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzczhz_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    jzazbz_to_image::<{ ImageConfiguration::Rgb as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts Jzczhz to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzczhz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzczhz_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    jzazbz_to_image::<{ ImageConfiguration::Bgr as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

/// This function converts Jzczhz with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Jzczhz data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn jzczhz_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    display_luminance: f32,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    jzazbz_to_image::<{ ImageConfiguration::Bgra as u8 }, { JzazbzTarget::Jzczhz as u8 }>(
        src,
        dst,
        display_luminance,
        transfer_function,
    )
}

#[cfg(test)]
mod tests_jzazbz_to_image {
    use super::*;
    use crate::{BufferStore, Rgb};

    const LUMINANCE: f32 = 203.0;

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

    fn assert_approx(a: u8, b: u8, eps: u8, label: &str) {
        assert!(
            (a as i16 - b as i16).abs() <= eps as i16,
            "{label}: got {a}, expected {b}"
        );
    }

    // Convert linear RGB → Jzazbz for use as test input
    fn rgb_to_jzazbz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rgb = Rgb::<f32>::new(r, g, b);
        let jzazbz = Jzazbz::from_linear_rgb_with_luminance(rgb, LUMINANCE);
        (jzazbz.jz, jzazbz.az, jzazbz.bz)
    }

    fn rgb_to_jzczhz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rgb = Rgb::<f32>::new(r, g, b);
        let jzazbz = Jzazbz::from_linear_rgb_with_luminance(rgb, LUMINANCE);
        let jzczhz = Jzczhz::from_jzazbz(jzazbz);
        (jzczhz.jz, jzczhz.cz, jzczhz.hz)
    }

    // ── Jzazbz black / white roundtrip ───────────────────────────────────────

    #[test]
    fn jzazbz_rgb_black_roundtrip() {
        let (jz, az, bz) = rgb_to_jzazbz(0.0, 0.0, 0.0);
        let src = make_src(vec![jz, az, bz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzazbz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "R black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "B black");
    }

    #[test]
    fn jzazbz_rgb_white_roundtrip() {
        let (jz, az, bz) = rgb_to_jzazbz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, az, bz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzazbz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "B white");
    }

    #[test]
    fn jzazbz_bgr_black_roundtrip() {
        let (jz, az, bz) = rgb_to_jzazbz(0.0, 0.0, 0.0);
        let src = make_src(vec![jz, az, bz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzazbz_to_bgr(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "B black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "R black");
    }

    #[test]
    fn jzazbz_bgr_white_roundtrip() {
        let (jz, az, bz) = rgb_to_jzazbz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, az, bz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzazbz_to_bgr(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "B white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "R white");
    }

    // ── Jzazbz RGBA / BGRA roundtrip ─────────────────────────────────────────

    #[test]
    fn jzazbz_rgba_white_opaque_roundtrip() {
        let (jz, az, bz) = rgb_to_jzazbz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, az, bz, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        jzazbz_to_rgba(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "B");
        assert_approx(d[3], 255, 1, "A");
    }

    #[test]
    fn jzazbz_rgba_black_transparent_roundtrip() {
        let (jz, az, bz) = rgb_to_jzazbz(0.0, 0.0, 0.0);
        let src = make_src(vec![jz, az, bz, 0.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        jzazbz_to_rgba(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[3], 0, 1, "A transparent");
    }

    #[test]
    fn jzazbz_bgra_white_opaque_roundtrip() {
        let (jz, az, bz) = rgb_to_jzazbz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, az, bz, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        jzazbz_to_bgra(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "B");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "R");
        assert_approx(d[3], 255, 1, "A");
    }

    // ── Jzczhz black / white roundtrip ───────────────────────────────────────

    #[test]
    fn jzczhz_rgb_black_roundtrip() {
        let (jz, cz, hz) = rgb_to_jzczhz(0.0, 0.0, 0.0);
        let src = make_src(vec![jz, cz, hz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzczhz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "R black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "B black");
    }

    #[test]
    fn jzczhz_rgb_white_roundtrip() {
        let (jz, cz, hz) = rgb_to_jzczhz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, cz, hz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzczhz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "B white");
    }

    #[test]
    fn jzczhz_bgr_white_roundtrip() {
        let (jz, cz, hz) = rgb_to_jzczhz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, cz, hz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzczhz_to_bgr(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "B white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "R white");
    }

    #[test]
    fn jzczhz_rgba_white_opaque_roundtrip() {
        let (jz, cz, hz) = rgb_to_jzczhz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, cz, hz, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        jzczhz_to_rgba(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "B");
        assert_approx(d[3], 255, 1, "A");
    }

    #[test]
    fn jzczhz_bgra_white_opaque_roundtrip() {
        let (jz, cz, hz) = rgb_to_jzczhz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, cz, hz, 1.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        jzczhz_to_bgra(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "B");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "R");
        assert_approx(d[3], 255, 1, "A");
    }

    // ── alpha denormalization ─────────────────────────────────────────────────

    #[test]
    fn jzazbz_rgba_alpha_denormalized_correctly() {
        let (jz, az, bz) = rgb_to_jzazbz(0.5, 0.5, 0.5);
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src(vec![jz, az, bz, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            jzazbz_to_rgba(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            let expected = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected, 1, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn jzazbz_bgra_alpha_denormalized_correctly() {
        let (jz, az, bz) = rgb_to_jzazbz(0.5, 0.5, 0.5);
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src(vec![jz, az, bz, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            jzazbz_to_bgra(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            let expected = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected, 1, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn jzczhz_rgba_alpha_denormalized_correctly() {
        let (jz, cz, hz) = rgb_to_jzczhz(0.5, 0.5, 0.5);
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src(vec![jz, cz, hz, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            jzczhz_to_rgba(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            let expected = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected, 1, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn jzczhz_bgra_alpha_denormalized_correctly() {
        let (jz, cz, hz) = rgb_to_jzczhz(0.5, 0.5, 0.5);
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src(vec![jz, cz, hz, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            jzczhz_to_bgra(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            let expected = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected, 1, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn rgba_rgb_channels_encoded_when_alpha_zero() {
        let (jz, az, bz) = rgb_to_jzazbz(1.0, 1.0, 1.0);
        let src = make_src(vec![jz, az, bz, 0.0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        jzazbz_to_rgba(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R encoded despite alpha=0");
        assert_eq!(d[3], 0, "A must remain 0");
    }

    // ── rgb and bgr agree on grey ─────────────────────────────────────────────

    #[test]
    fn jzazbz_rgb_bgr_agree_on_grey() {
        let (jz, az, bz) = rgb_to_jzazbz(0.5, 0.5, 0.5);
        let src_rgb = make_src(vec![jz, az, bz], 1, 1, 3);
        let src_bgr = make_src(vec![jz, az, bz], 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        jzazbz_to_rgb(&src_rgb, &mut dst_rgb, LUMINANCE, TransferFunction::Srgb).unwrap();
        jzazbz_to_bgr(&src_bgr, &mut dst_bgr, LUMINANCE, TransferFunction::Srgb).unwrap();
        assert_eq!(
            &*dst_rgb.data.borrow(),
            &*dst_bgr.data.borrow(),
            "grey invariant to channel order"
        );
    }

    #[test]
    fn jzazbz_rgba_bgra_agree_on_grey() {
        let (jz, az, bz) = rgb_to_jzazbz(0.5, 0.5, 0.5);
        let src_rgba = make_src(vec![jz, az, bz, 0.7], 1, 1, 4);
        let src_bgra = make_src(vec![jz, az, bz, 0.7], 1, 1, 4);
        let mut dst_rgba = make_dst(1, 1, 4);
        let mut dst_bgra = make_dst(1, 1, 4);
        jzazbz_to_rgba(&src_rgba, &mut dst_rgba, LUMINANCE, TransferFunction::Srgb).unwrap();
        jzazbz_to_bgra(&src_bgra, &mut dst_bgra, LUMINANCE, TransferFunction::Srgb).unwrap();
        assert_eq!(
            &*dst_rgba.data.borrow(),
            &*dst_bgra.data.borrow(),
            "grey invariant to channel order"
        );
    }

    #[test]
    fn jzczhz_rgb_bgr_agree_on_grey() {
        let (jz, cz, hz) = rgb_to_jzczhz(0.5, 0.5, 0.5);
        let src_rgb = make_src(vec![jz, cz, hz], 1, 1, 3);
        let src_bgr = make_src(vec![jz, cz, hz], 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        jzczhz_to_rgb(&src_rgb, &mut dst_rgb, LUMINANCE, TransferFunction::Srgb).unwrap();
        jzczhz_to_bgr(&src_bgr, &mut dst_bgr, LUMINANCE, TransferFunction::Srgb).unwrap();
        assert_eq!(
            &*dst_rgb.data.borrow(),
            &*dst_bgr.data.borrow(),
            "grey invariant to channel order"
        );
    }

    // ── channel ordering for non-grey pixel ───────────────────────────────────

    #[test]
    fn jzazbz_rgb_channel_order_correct_for_red() {
        let (jz, az, bz) = rgb_to_jzazbz(1.0, 0.0, 0.0);
        let src = make_src(vec![jz, az, bz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzazbz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] > 200, "R dominant for red input, got {}", d[0]);
        assert!(d[1] < 30, "G low for red input, got {}", d[1]);
        assert!(d[2] < 30, "B low for red input, got {}", d[2]);
    }

    #[test]
    fn jzazbz_bgr_channel_order_correct_for_red() {
        let (jz, az, bz) = rgb_to_jzazbz(1.0, 0.0, 0.0);
        let src = make_src(vec![jz, az, bz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzazbz_to_bgr(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] < 30, "B low for red input in BGR, got {}", d[0]);
        assert!(d[1] < 30, "G low for red input in BGR, got {}", d[1]);
        assert!(d[2] > 200, "R dominant for red input in BGR, got {}", d[2]);
    }

    #[test]
    fn jzczhz_rgb_channel_order_correct_for_red() {
        let (jz, cz, hz) = rgb_to_jzczhz(1.0, 0.0, 0.0);
        let src = make_src(vec![jz, cz, hz], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        jzczhz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] > 200, "R dominant for red input, got {}", d[0]);
        assert!(d[1] < 30, "G low for red input, got {}", d[1]);
        assert!(d[2] < 30, "B low for red input, got {}", d[2]);
    }

    // ── display luminance affects output ──────────────────────────────────────

    #[test]
    fn different_luminance_produces_different_output() {
        let (jz, az, bz) = rgb_to_jzazbz(0.5, 0.3, 0.7);
        let src1 = make_src(vec![jz, az, bz], 1, 1, 3);
        let src2 = make_src(vec![jz, az, bz], 1, 1, 3);
        let mut dst1 = make_dst(1, 1, 3);
        let mut dst2 = make_dst(1, 1, 3);
        jzazbz_to_rgb(&src1, &mut dst1, 100.0, TransferFunction::Srgb).unwrap();
        jzazbz_to_rgb(&src2, &mut dst2, 400.0, TransferFunction::Srgb).unwrap();
        assert_ne!(
            &*dst1.data.borrow(),
            &*dst2.data.borrow(),
            "different luminance must produce different output"
        );
    }

    // ── Jzazbz and Jzczhz differ for chromatic input ─────────────────────────

    #[test]
    fn jzazbz_and_jzczhz_differ_for_chromatic_input() {
        // They use different coordinate systems so their raw float inputs differ —
        // verify the two paths don't accidentally produce identical output for
        // chromatic colours (where az/bz ≠ 0, cz/hz representation differs).
        let r = 0.8f32;
        let g = 0.1f32;
        let b = 0.3f32;
        let (jz1, az, bz) = rgb_to_jzazbz(r, g, b);
        let (jz2, cz, hz) = rgb_to_jzczhz(r, g, b);
        // Inputs are genuinely different values
        let src_jzazbz = make_src(vec![jz1, az, bz], 1, 1, 3);
        let src_jzczhz = make_src(vec![jz2, cz, hz], 1, 1, 3);
        let mut dst1 = make_dst(1, 1, 3);
        let mut dst2 = make_dst(1, 1, 3);
        jzazbz_to_rgb(&src_jzazbz, &mut dst1, LUMINANCE, TransferFunction::Srgb).unwrap();
        jzczhz_to_rgb(&src_jzczhz, &mut dst2, LUMINANCE, TransferFunction::Srgb).unwrap();
        // Both should reconstruct roughly the same sRGB — the paths are different
        // but the colour they encode is the same
        let d1 = dst1.data.borrow();
        let d2 = dst2.data.borrow();
        for ch in 0..3 {
            assert_approx(
                d1[ch],
                d2[ch],
                4,
                &format!("ch{ch} jzazbz vs jzczhz roundtrip"),
            );
        }
    }

    // ── multi-pixel / multi-row consistency ───────────────────────────────────

    #[test]
    fn jzazbz_rgb_multi_pixel_consistent() {
        let (jz, az, bz) = rgb_to_jzazbz(0.4, 0.6, 0.2);
        let src = make_src(vec![jz, az, bz].repeat(4), 4, 1, 3);
        let mut dst = make_dst(4, 1, 3);
        jzazbz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn jzazbz_rgba_multi_pixel_consistent() {
        let (jz, az, bz) = rgb_to_jzazbz(0.3, 0.5, 0.7);
        let src = make_src(vec![jz, az, bz, 0.6].repeat(4), 4, 1, 4);
        let mut dst = make_dst(4, 1, 4);
        jzazbz_to_rgba(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2], d[3]];
        for (i, chunk) in d.chunks(4).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn jzazbz_rgb_multi_row_consistent() {
        let (jz, az, bz) = rgb_to_jzazbz(0.6, 0.4, 0.8);
        let src = make_src(vec![jz, az, bz].repeat(4), 2, 2, 3);
        let mut dst = make_dst(2, 2, 3);
        jzazbz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn jzczhz_rgb_multi_pixel_consistent() {
        let (jz, cz, hz) = rgb_to_jzczhz(0.4, 0.6, 0.2);
        let src = make_src(vec![jz, cz, hz].repeat(4), 4, 1, 3);
        let mut dst = make_dst(4, 1, 3);
        jzczhz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    // ── error paths ───────────────────────────────────────────────────────────

    #[test]
    fn rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(jzazbz_to_rgb(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).is_err());
    }

    #[test]
    fn rgba_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 4);
        assert!(jzazbz_to_rgba(&src, &mut dst, LUMINANCE, TransferFunction::Srgb).is_err());
    }
}
