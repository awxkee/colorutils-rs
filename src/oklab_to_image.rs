/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_oklab_to_image;
use crate::image::ImageConfiguration;
use crate::image_to_oklab::OklabTarget;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_oklab_to_image;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_oklab_to_image;
use crate::{ColorError, ImageBuffer, ImageBufferMut, Oklab, Oklch, Rgb, TransferFunction};

fn oklab_to_image<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
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
    let target: OklabTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();

    let mut _wide_row_handle: Option<unsafe fn(usize, &[f32], &mut [f32], u32) -> usize> = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_oklab_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if is_x86_feature_detected!("avx2") {
        _wide_row_handle = Some(avx_oklab_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handle = Some(neon_oklab_to_image::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let mut lut_table = [0u8; 65535];
    for (i, lut) in lut_table[..2049].iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.)
            .round()
            .min(255.) as u8;
    }

    let channels = image_configuration.get_channels_count();

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;
    let width = src.width;

    let mut transient_row = vec![0f32; src.width as usize * channels];

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
            cx = unsafe { dispatcher(cx, src, &mut transient_row, width) };
        }

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            let transient_row = &mut transient_row[cx * 3..];
            let src = &src[cx * 3usize..];
            for (dst, src) in transient_row
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                let rgb = match target {
                    OklabTarget::Oklab => {
                        let oklab = Oklab::new(src[0], src[1], src[2]);
                        oklab.to_linear_rgb()
                    }
                    OklabTarget::Oklch => {
                        let oklch = Oklch::new(src[0], src[1], src[2]);
                        oklch.to_linear_rgb()
                    }
                };
                dst[0] = rgb.r;
                dst[1] = rgb.g;
                dst[2] = rgb.b;
            }

            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(transient_row.as_chunks::<3>().0.iter())
            {
                let rgb = (Rgb::<f32>::new(
                    src[0].max(0.).min(1.),
                    src[1].max(0.).min(1.),
                    src[2].max(0.).min(1.),
                ) * Rgb::<f32>::dup(2048f32))
                .round()
                .cast::<u16>();

                dst[image_configuration.r_index()] = lut_table[rgb.r.min(2048) as usize];
                dst[image_configuration.g_index()] = lut_table[rgb.g.min(2048) as usize];
                dst[image_configuration.b_index()] = lut_table[rgb.b.min(2048) as usize];
            }
        } else if image_configuration == ImageConfiguration::Bgra
            || image_configuration == ImageConfiguration::Rgba
        {
            let transient_row = &mut transient_row[cx * 4..];
            let src = &src[cx * 4usize..];
            for (dst, src) in transient_row
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                let rgb = match target {
                    OklabTarget::Oklab => {
                        let oklab = Oklab::new(src[0], src[1], src[2]);
                        oklab.to_linear_rgb()
                    }
                    OklabTarget::Oklch => {
                        let oklch = Oklch::new(src[0], src[1], src[2]);
                        oklch.to_linear_rgb()
                    }
                };
                dst[0] = rgb.r;
                dst[1] = rgb.g;
                dst[2] = rgb.b;
                dst[3] = src[3];
            }

            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(transient_row.as_chunks::<4>().0.iter())
            {
                let rgb = (Rgb::<f32>::new(
                    src[0].max(0.).min(1.),
                    src[1].max(0.).min(1.),
                    src[2].max(0.).min(1.),
                ) * Rgb::<f32>::dup(2048f32))
                .round()
                .cast::<u16>();

                dst[image_configuration.r_index()] = lut_table[rgb.r.min(2048) as usize];
                dst[image_configuration.g_index()] = lut_table[rgb.g.min(2048) as usize];
                dst[image_configuration.b_index()] = lut_table[rgb.b.min(2048) as usize];
                dst[image_configuration.a_index()] = (src[3] * 255.).round() as u8;
            }
        }
    }

    Ok(())
}

/// This function converts Oklab with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklab_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    oklab_to_image::<{ ImageConfiguration::Rgba as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts Oklab to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklab_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    oklab_to_image::<{ ImageConfiguration::Rgb as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts Oklab to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklab_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    oklab_to_image::<{ ImageConfiguration::Bgr as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts Oklab with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklab_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    oklab_to_image::<{ ImageConfiguration::Bgra as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts *Oklch* with interleaved alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCH data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGBA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklch_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    oklab_to_image::<{ ImageConfiguration::Rgba as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts *Oklch* to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCH data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklch_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    oklab_to_image::<{ ImageConfiguration::Rgb as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts *Oklch* to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCH data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklch_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    oklab_to_image::<{ ImageConfiguration::Bgr as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts *Oklch* with interleaved alpha channel to BGRA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LCH data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive BGRA data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from linear colorspace to gamma
pub fn oklch_to_bgra(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    oklab_to_image::<{ ImageConfiguration::Bgra as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BufferStore;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_src(data: Vec<f32>, width: u32, channels: u32) -> ImageBuffer<'static, f32> {
        ImageBuffer {
            data: std::borrow::Cow::Owned(data),
            width,
            height: 1,
            channels,
            stride: width * channels,
        }
    }

    fn make_dst(width: u32, channels: u32) -> ImageBufferMut<'static, u8> {
        ImageBufferMut {
            data: BufferStore::Owned(vec![0u8; (width * channels) as usize]),
            width,
            height: 1,
            channels,
            stride: width * channels,
        }
    }

    // ── oklab_to_rgb ─────────────────────────────────────────────────────────

    #[test]
    fn test_oklab_to_rgb_white() {
        // Oklab white: L=1.0, a=0.0, b=0.0 => RGB (255,255,255)
        let src = make_src(vec![1.0f32, 0.0, 0.0], 1, 3);
        let mut dst = make_dst(1, 3);
        oklab_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data[0], 255, "R should be 255 for white");
        assert_eq!(dst.data[1], 255, "G should be 255 for white");
        assert_eq!(dst.data[2], 255, "B should be 255 for white");
    }

    #[test]
    fn test_oklab_to_rgb_black() {
        // Oklab black: L=0.0, a=0.0, b=0.0 => RGB (0,0,0)
        let src = make_src(vec![0.0f32, 0.0, 0.0], 1, 3);
        let mut dst = make_dst(1, 3);
        oklab_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data[0], 0);
        assert_eq!(dst.data[1], 0);
        assert_eq!(dst.data[2], 0);
    }

    #[test]
    fn test_oklab_to_rgb_multiple_pixels() {
        // Two pixels: black and white
        let src = make_src(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 3);
        let mut dst = make_dst(2, 3);
        oklab_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&dst.data[0..3], &[0, 0, 0]);
        assert_eq!(&dst.data[3..6], &[255, 255, 255]);
    }

    // ── oklab_to_rgba ────────────────────────────────────────────────────────

    #[test]
    fn test_oklab_to_rgba_white_full_alpha() {
        // L=1, a=0, b=0, alpha=1.0 => (255,255,255,255)
        let src = make_src(vec![1.0f32, 0.0, 0.0, 1.0], 1, 4);
        let mut dst = make_dst(1, 4);
        oklab_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data[0], 255);
        assert_eq!(dst.data[1], 255);
        assert_eq!(dst.data[2], 255);
        assert_eq!(dst.data[3], 255, "Alpha should be 255");
    }

    #[test]
    fn test_oklab_to_rgba_zero_alpha() {
        let src = make_src(vec![1.0f32, 0.0, 0.0, 0.0], 1, 4);
        let mut dst = make_dst(1, 4);
        oklab_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data[3], 0, "Alpha should be 0");
    }

    #[test]
    fn test_oklab_to_rgba_half_alpha() {
        let src = make_src(vec![1.0f32, 0.0, 0.0, 0.5], 1, 4);
        let mut dst = make_dst(1, 4);
        oklab_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let alpha = dst.data[3];
        assert!((alpha as i16 - 128).abs() <= 2, "Alpha ~128, got {alpha}");
    }

    // ── oklab_to_bgr ─────────────────────────────────────────────────────────

    #[test]
    fn test_oklab_to_bgr_white() {
        let src = make_src(vec![1.0f32, 0.0, 0.0], 1, 3);
        let mut dst = make_dst(1, 3);
        oklab_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
        // BGR order — all 255 for white
        assert_eq!(dst.data[0], 255);
        assert_eq!(dst.data[1], 255);
        assert_eq!(dst.data[2], 255);
    }

    // ── oklab_to_bgra ────────────────────────────────────────────────────────

    #[test]
    fn test_oklab_to_bgra_alpha_preserved() {
        let src = make_src(vec![0.5f32, 0.0, 0.0, 0.75], 1, 4);
        let mut dst = make_dst(1, 4);
        oklab_to_bgra(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let alpha = dst.data[3];
        assert!((alpha as i16 - 191).abs() <= 2, "Alpha ~191, got {alpha}");
    }

    // ── oklch_to_rgb ─────────────────────────────────────────────────────────

    #[test]
    fn test_oklch_to_rgb_white() {
        // Oklch white: L=1.0, C=0.0, H=0.0 => RGB (255,255,255)
        let src = make_src(vec![1.0f32, 0.0, 0.0], 1, 3);
        let mut dst = make_dst(1, 3);
        oklch_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data[0], 255);
        assert_eq!(dst.data[1], 255);
        assert_eq!(dst.data[2], 255);
    }

    #[test]
    fn test_oklch_to_rgb_black() {
        let src = make_src(vec![0.0f32, 0.0, 0.0], 1, 3);
        let mut dst = make_dst(1, 3);
        oklch_to_rgb(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&dst.data[0..3], &[0, 0, 0]);
    }

    // ── oklch_to_rgba ────────────────────────────────────────────────────────

    #[test]
    fn test_oklch_to_rgba_alpha_passthrough() {
        let src = make_src(vec![1.0f32, 0.0, 0.0, 0.5], 1, 4);
        let mut dst = make_dst(1, 4);
        oklch_to_rgba(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let alpha = dst.data[3];
        assert!((alpha as i16 - 128).abs() <= 2, "Alpha ~128, got {alpha}");
    }

    // ── oklch_to_bgr / oklch_to_bgra ────────────────────────────────────────

    #[test]
    fn test_oklch_to_bgr_white() {
        let src = make_src(vec![1.0f32, 0.0, 0.0], 1, 3);
        let mut dst = make_dst(1, 3);
        oklch_to_bgr(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&dst.data[0..3], &[255, 255, 255]);
    }

    #[test]
    fn test_oklch_to_bgra_black_zero_alpha() {
        let src = make_src(vec![0.0f32, 0.0, 0.0, 0.0], 1, 4);
        let mut dst = make_dst(1, 4);
        oklch_to_bgra(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(&dst.data[0..4], &[0, 0, 0, 0]);
    }

    // ── error / edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_invalid_channel_count_returns_error() {
        let src = make_src(vec![1.0f32, 0.0], 1, 2); // 2 channels — unsupported
        let mut dst = make_dst(1, 2);
        let result = oklab_to_rgb(&src, &mut dst, TransferFunction::Srgb);
        assert!(result.is_err(), "Should fail with 2 channels");
    }

    #[test]
    fn test_out_of_range_values_clamped() {
        // Values > 1.0 should clamp to 255, not panic
        let src = make_src(vec![2.0f32, 0.5, 0.5], 1, 3);
        let mut dst = make_dst(1, 3);
        let result = oklab_to_rgb(&src, &mut dst, TransferFunction::Srgb);
        assert!(result.is_ok());
        assert_eq!(dst.data[0], 255);
    }

    #[test]
    fn test_negative_values_clamped() {
        let src = make_src(vec![-1.0f32, 0.0, 0.0], 1, 3);
        let mut dst = make_dst(1, 3);
        let result = oklab_to_rgb(&src, &mut dst, TransferFunction::Srgb);
        assert!(result.is_ok());
    }

    #[test]
    fn test_linear_transfer_function() {
        let src = make_src(vec![1.0f32, 0.0, 0.0], 1, 3);
        let mut dst = make_dst(1, 3);
        oklab_to_rgb(&src, &mut dst, TransferFunction::Linear).unwrap();
        assert_eq!(&dst.data[0..3], &[255, 255, 255]);
    }
}
