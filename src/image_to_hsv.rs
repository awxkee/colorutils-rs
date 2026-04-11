/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_channels_to_hsv_u16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_channels_to_hsv_u16;
use crate::{ColorError, ImageBuffer, ImageBufferMut, Rgb};

#[allow(clippy::type_complexity)]
fn channels_to_hsv_u16<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u16>,
    scale: f32,
) -> Result<(), ColorError> {
    src.validate()?;
    dst.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    let target: HsvTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut _wide_row_handler: Option<unsafe fn(usize, &[u8], u32, &mut [u16], f32) -> usize> =
        None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler =
            Some(neon_channels_to_hsv_u16::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler =
            Some(sse_channels_to_hsv_u16::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
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
            cx = unsafe { dispatcher(cx, src, width, dst, scale) };
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
                let r = src[image_configuration.r_index()];
                let g = src[image_configuration.g_index()];
                let b = src[image_configuration.b_index()];
                let rgb = Rgb::<u8>::new(r, g, b);
                match target {
                    HsvTarget::Hsv => {
                        let hsv = rgb.to_hsv();

                        dst[0] = hsv.h as u16;
                        dst[1] = (hsv.s * scale).round() as u16;
                        dst[2] = (hsv.v * scale).round() as u16;
                    }
                    HsvTarget::Hsl => {
                        let hsl = rgb.to_hsl();

                        dst[0] = hsl.h as u16;
                        dst[1] = (hsl.s * scale).round() as u16;
                        dst[2] = (hsl.l * scale).round() as u16;
                    }
                }
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
                let r = src[image_configuration.r_index()];
                let g = src[image_configuration.g_index()];
                let b = src[image_configuration.b_index()];
                let rgb = Rgb::<u8>::new(r, g, b);
                match target {
                    HsvTarget::Hsv => {
                        let hsv = rgb.to_hsv();

                        dst[0] = hsv.h as u16;
                        dst[1] = (hsv.s * scale).round() as u16;
                        dst[2] = (hsv.v * scale).round() as u16;
                    }
                    HsvTarget::Hsl => {
                        let hsl = rgb.to_hsl();

                        dst[0] = hsl.h as u16;
                        dst[1] = (hsl.s * scale).round() as u16;
                        dst[2] = (hsl.l * scale).round() as u16;
                    }
                }
                dst[3] = src[image_configuration.a_index()] as u16;
            }
        }
    }
    Ok(())
}

/// This function converts RGB to HSV. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn rgb_to_hsv(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u16>,
    scale: f32,
) -> Result<(), ColorError> {
    channels_to_hsv_u16::<{ ImageConfiguration::Rgb as u8 }, false, { HsvTarget::Hsv as u8 }>(
        src, dst, scale,
    )
}

/// This function converts BGRA to HSV. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn bgra_to_hsv(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u16>,
    scale: f32,
) -> Result<(), ColorError> {
    channels_to_hsv_u16::<{ ImageConfiguration::Bgra as u8 }, true, { HsvTarget::Hsv as u8 }>(
        src, dst, scale,
    )
}

/// This function converts RGBA to HSV. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSV data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn rgba_to_hsv(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u16>,
    scale: f32,
) -> Result<(), ColorError> {
    channels_to_hsv_u16::<{ ImageConfiguration::Rgba as u8 }, true, { HsvTarget::Hsv as u8 }>(
        src, dst, scale,
    )
}

/// This function converts RGB to HSL. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSL data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and L is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,L in range [0, scale]
pub fn rgb_to_hsl(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u16>,
    scale: f32,
) -> Result<(), ColorError> {
    channels_to_hsv_u16::<{ ImageConfiguration::Rgb as u8 }, false, { HsvTarget::Hsl as u8 }>(
        src, dst, scale,
    )
}

/// This function converts BGRA to HSL. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSL data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn bgra_to_hsl(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u16>,
    scale: f32,
) -> Result<(), ColorError> {
    channels_to_hsv_u16::<{ ImageConfiguration::Bgra as u8 }, true, { HsvTarget::Hsl as u8 }>(
        src, dst, scale,
    )
}

/// This function converts RGBA to HSL. Alpha channel is copied and leaved unchanged. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive HSL data
/// * `dst_stride` - Bytes per row for dst data
/// * `scale` - Natural range for S and V is [0,1] it may be more convenient and required for u16 transformation to scale it by 100 or any other number to keep S,V in range [0, scale]
pub fn rgba_to_hsl(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, u16>,
    scale: f32,
) -> Result<(), ColorError> {
    channels_to_hsv_u16::<{ ImageConfiguration::Rgba as u8 }, true, { HsvTarget::Hsl as u8 }>(
        src, dst, scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BufferStore;
    use std::borrow::Cow;

    fn make_src(data: &[u8], width: u32, height: u32, channels: u32) -> ImageBuffer<'static, u8> {
        ImageBuffer {
            data: Cow::Owned(data.to_vec()),
            width,
            height,
            channels,
            stride: width * channels,
        }
    }

    fn make_dst(
        data: &[u16],
        width: u32,
        height: u32,
        channels: u32,
    ) -> ImageBufferMut<'static, u16> {
        ImageBufferMut {
            data: BufferStore::Owned(data.to_vec()),
            width,
            height,
            channels,
            stride: width * channels,
        }
    }

    #[test]
    fn test_rgb_to_hsv_red() {
        let src_data = [255u8, 0, 0];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsv(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[0], 0); // H = 0
        assert_eq!(dst.data[1], 100); // S = 100%
        assert_eq!(dst.data[2], 100); // V = 100%
    }

    #[test]
    fn test_rgb_to_hsv_green() {
        let src_data = [0u8, 255, 0];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsv(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[0], 120);
        assert_eq!(dst.data[1], 100);
        assert_eq!(dst.data[2], 100);
    }

    #[test]
    fn test_rgb_to_hsv_blue() {
        let src_data = [0u8, 0, 255];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsv(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[0], 240); // H = 240
        assert_eq!(dst.data[1], 100);
        assert_eq!(dst.data[2], 100);
    }

    #[test]
    fn test_rgb_to_hsv_white() {
        let src_data = [255u8, 255, 255];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsv(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[1], 0);
        assert_eq!(dst.data[2], 100);
    }

    #[test]
    fn test_rgb_to_hsv_black() {
        let src_data = [0u8, 0, 0];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsv(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[1], 0); // S = 0
        assert_eq!(dst.data[2], 0); // V = 0
    }

    #[test]
    fn test_rgb_to_hsv_scale() {
        let src_data = [255u8, 0, 0];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsv(&src, &mut dst, 255.0).unwrap();
        assert_eq!(dst.data[1], 255); // S scaled to 255
        assert_eq!(dst.data[2], 255); // V scaled to 255
    }

    #[test]
    fn test_rgba_to_hsv_alpha_preserved() {
        let src_data = [255u8, 0, 0, 128]; // red with alpha=128
        let mut dst_data = [0u16; 4];
        let src = make_src(&src_data, 1, 1, 4);
        let mut dst = make_dst(&mut dst_data, 1, 1, 4);
        rgba_to_hsv(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[0], 0); // H
        assert_eq!(dst.data[1], 100); // S
        assert_eq!(dst.data[2], 100); // V
        assert_eq!(dst.data[3], 128);
    }

    #[test]
    fn test_rgb_to_hsl_red() {
        let src_data = [255u8, 0, 0];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsl(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[0], 0);
        assert_eq!(dst.data[1], 100);
        assert_eq!(dst.data[2], 50);
    }

    #[test]
    fn test_rgb_to_hsl_white() {
        let src_data = [255u8, 255, 255];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsl(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[1], 0); // S = 0
        assert_eq!(dst.data[2], 100); // L = 100%
    }

    #[test]
    fn test_rgb_to_hsl_black() {
        let src_data = [0u8, 0, 0];
        let mut dst_data = [0u16; 3];
        let src = make_src(&src_data, 1, 1, 3);
        let mut dst = make_dst(&mut dst_data, 1, 1, 3);
        rgb_to_hsl(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[1], 0); // S = 0
        assert_eq!(dst.data[2], 0); // L = 0
    }

    #[test]
    fn test_rgba_to_hsl_alpha_preserved() {
        let src_data = [0u8, 255, 0, 200];
        let mut dst_data = [0u16; 4];
        let src = make_src(&src_data, 1, 1, 4);
        let mut dst = make_dst(&mut dst_data, 1, 1, 4);
        rgba_to_hsl(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[0], 120);
        assert_eq!(dst.data[3], 200);
    }

    #[test]
    fn test_rgb_to_hsv_multi_pixel() {
        // red, green, blue side by side
        let src_data = [255u8, 0, 0, 0, 255, 0, 0, 0, 255];
        let mut dst_data = [0u16; 9];
        let src = make_src(&src_data, 3, 1, 3);
        let mut dst = make_dst(&mut dst_data, 3, 1, 3);
        rgb_to_hsv(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[0], 0); // red H
        assert_eq!(dst.data[3], 120); // green H
        assert_eq!(dst.data[6], 240); // blue H
    }

    // --- bgra_to_hsv ---

    #[test]
    fn test_bgra_to_hsv_red() {
        let src_data = [0u8, 0, 255, 255];
        let mut dst_data = [0u16; 4];
        let src = make_src(&src_data, 1, 1, 4);
        let mut dst = make_dst(&mut dst_data, 1, 1, 4);
        bgra_to_hsv(&src, &mut dst, 100.0).unwrap();
        assert_eq!(dst.data[0], 0); // H = 0 (red)
        assert_eq!(dst.data[1], 100); // S
        assert_eq!(dst.data[2], 100); // V
        assert_eq!(dst.data[3], 255); // alpha
    }
}
