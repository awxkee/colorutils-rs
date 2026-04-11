/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_image_to_oklab;
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_image_to_oklab;
use crate::oklch::Oklch;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_image_to_oklab;
use crate::{
    ColorError, ImageBuffer, ImageBufferMut, Oklab, Rgb, TransferFunction, bgr_to_linear,
    bgra_to_linear, rgb_to_linear, rgba_to_linear,
};

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum OklabTarget {
    Oklab = 0,
    Oklch = 1,
}

impl From<u8> for OklabTarget {
    fn from(value: u8) -> Self {
        match value {
            0 => OklabTarget::Oklab,
            1 => OklabTarget::Oklch,
            _ => {
                panic!("Not implemented")
            }
        }
    }
}

fn channels_to_oklab<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
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

    let linear_converter = match image_configuration {
        ImageConfiguration::Rgb => rgb_to_linear,
        ImageConfiguration::Rgba => rgba_to_linear,
        ImageConfiguration::Bgra => bgra_to_linear,
        ImageConfiguration::Bgr => bgr_to_linear,
    };

    linear_converter(src, dst, transfer_function)?;

    let mut _wide_row_handle: Option<unsafe fn(usize, u32, &mut [f32]) -> usize> = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handle = Some(neon_image_to_oklab::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handle = Some(sse_image_to_oklab::<CHANNELS_CONFIGURATION, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handle = Some(avx_image_to_oklab::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let dst_stride = dst.stride();
    let dst_full_width = dst.width
        * if image_configuration.has_alpha() {
            4
        } else {
            3
        };
    let width = dst.width;
    for dst_row in dst.data.borrow_mut().chunks_mut(dst_stride) {
        let dst = &mut dst_row[..dst_full_width as usize];

        let mut cx = 0usize;

        if let Some(dispatcher) = _wide_row_handle {
            cx = unsafe { dispatcher(cx, width, dst) };
        }

        let rem = &mut dst_row[cx * image_configuration.channel_count()..];
        if image_configuration.has_alpha() {
            for dst in rem.as_chunks_mut::<4>().0.iter_mut() {
                let rgb = Rgb::<f32>::new(
                    dst[image_configuration.r_index()],
                    dst[image_configuration.g_index()],
                    dst[image_configuration.b_index()],
                );
                match target {
                    OklabTarget::Oklab => {
                        let oklab = Oklab::from_linear_rgb(rgb);
                        dst[0] = oklab.l;
                        dst[1] = oklab.a;
                        dst[2] = oklab.b;
                    }
                    OklabTarget::Oklch => {
                        let oklch = Oklch::from_linear_rgb(rgb);
                        dst[0] = oklch.l;
                        dst[1] = oklch.c;
                        dst[2] = oklch.c;
                    }
                }
            }
        } else {
            for dst in rem.as_chunks_mut::<3>().0.iter_mut() {
                let rgb = Rgb::<f32>::new(
                    dst[image_configuration.r_index()],
                    dst[image_configuration.g_index()],
                    dst[image_configuration.b_index()],
                );
                match target {
                    OklabTarget::Oklab => {
                        let oklab = Oklab::from_linear_rgb(rgb);
                        dst[0] = oklab.l;
                        dst[1] = oklab.a;
                        dst[2] = oklab.b;
                    }
                    OklabTarget::Oklch => {
                        let oklch = Oklch::from_linear_rgb(rgb);
                        dst[0] = oklch.l;
                        dst[1] = oklch.c;
                        dst[2] = oklch.h;
                    }
                }
            }
        }
    }
    Ok(())
}

/// This function converts RGB to Oklab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_oklab(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_oklab::<{ ImageConfiguration::Rgb as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts RGBA to Oklab against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_oklab(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_oklab::<{ ImageConfiguration::Rgba as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts BGRA to Oklab against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_oklab(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_oklab::<{ ImageConfiguration::Bgra as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts BGR to Oklab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_oklab(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_oklab::<{ ImageConfiguration::Bgr as u8 }, { OklabTarget::Oklab as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts RGB to Oklch against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgb_to_oklch(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_oklab::<{ ImageConfiguration::Rgb as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts RGBA to Oklch against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn rgba_to_oklch(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_oklab::<{ ImageConfiguration::Rgba as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts BGRA to Oklch against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgra_to_oklch(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_oklab::<{ ImageConfiguration::Bgra as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        dst,
        transfer_function,
    )
}

/// This function converts BGR to Oklch against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - transfer function to linear colorspace
pub fn bgr_to_oklch(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_oklab::<{ ImageConfiguration::Bgr as u8 }, { OklabTarget::Oklch as u8 }>(
        src,
        dst,
        transfer_function,
    )
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

    struct OklabRef {
        rgb: [u8; 3],
        l: f32,
        a: f32,
        b: f32,
    }

    const OKLAB_REFS: &[OklabRef] = &[
        OklabRef {
            rgb: [0, 0, 0],
            l: 0.0,
            a: 0.0,
            b: 0.0,
        }, // black
        OklabRef {
            rgb: [255, 255, 255],
            l: 1.0,
            a: 0.0,
            b: 0.0,
        }, // white
        OklabRef {
            rgb: [255, 0, 0],
            l: 0.6279,
            a: 0.2249,
            b: 0.1257,
        }, // red
        OklabRef {
            rgb: [0, 255, 0],
            l: 0.8664,
            a: -0.2338,
            b: 0.1794,
        }, // green
        OklabRef {
            rgb: [0, 0, 255],
            l: 0.4520,
            a: -0.0324,
            b: -0.3118,
        }, // blue
    ];

    // ── validation ────────────────────────────────────────────────────────────

    #[test]
    fn unsupported_channel_count_returns_error() {
        // 2-channel images are not supported
        let src = make_src(1, 1, 2, vec![128u8, 128]);
        let mut dst = make_dst(1, 1, 2);
        let result = channels_to_oklab::<
            { ImageConfiguration::Rgb as u8 },
            { OklabTarget::Oklab as u8 },
        >(&src, &mut dst, TransferFunction::Srgb);
        assert!(matches!(
            result,
            Err(ColorError::UnsupportedChannelsCount(2))
        ));
    }

    #[test]
    fn mismatched_dimensions_returns_error() {
        let src = make_src(2, 2, 3, vec![0u8; 12]);
        let mut dst = make_dst(4, 4, 3);
        let result = rgb_to_oklab(&src, &mut dst, TransferFunction::Srgb);
        assert!(result.is_err());
    }

    // ── black / white anchors ─────────────────────────────────────────────────

    #[test]
    fn black_maps_to_oklab_zero() {
        let src = make_src(1, 1, 3, vec![0, 0, 0]);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_oklab(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d[0].abs() < 0.005, "L of black should be ≈ 0, got {}", d[0]);
        assert!(d[1].abs() < 0.005, "a of black should be ≈ 0, got {}", d[1]);
        assert!(d[2].abs() < 0.005, "b of black should be ≈ 0, got {}", d[2]);
    }

    #[test]
    fn white_maps_to_oklab_l1_ab0() {
        let src = make_src(1, 1, 3, vec![255, 255, 255]);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_oklab(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(
            (d[0] - 1.0).abs() < 0.005,
            "L of white should be ≈ 1, got {}",
            d[0]
        );
        assert!(d[1].abs() < 0.005, "a of white should be ≈ 0, got {}", d[1]);
        assert!(d[2].abs() < 0.005, "b of white should be ≈ 0, got {}", d[2]);
    }

    // ── reference values for primary colours ─────────────────────────────────

    #[test]
    fn primary_colours_match_oklab_reference_values() {
        for r in OKLAB_REFS {
            let src = make_src(1, 1, 3, r.rgb.to_vec());
            let mut dst = make_dst(1, 1, 3);
            rgb_to_oklab(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let d = dst.data.borrow();
            let tol = 0.005;
            assert!(
                (d[0] - r.l).abs() < tol,
                "rgb={:?}: L expected {}, got {}",
                r.rgb,
                r.l,
                d[0]
            );
            assert!(
                (d[1] - r.a).abs() < tol,
                "rgb={:?}: a expected {}, got {}",
                r.rgb,
                r.a,
                d[1]
            );
            assert!(
                (d[2] - r.b).abs() < tol,
                "rgb={:?}: b expected {}, got {}",
                r.rgb,
                r.b,
                d[2]
            );
        }
    }

    #[test]
    fn rgba_alpha_fully_opaque_is_one() {
        let src = make_src(1, 1, 4, vec![128, 64, 32, 255]);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_oklab(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let alpha = dst.data.borrow()[3];
        assert!((alpha - 1.0).abs() < 1e-4, "alpha 255 → 1.0, got {alpha}");
    }

    #[test]
    fn rgba_alpha_fully_transparent_is_zero() {
        let src = make_src(1, 1, 4, vec![128, 64, 32, 0]);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_oklab(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let alpha = dst.data.borrow()[3];
        assert_eq!(alpha, 0.0, "alpha 0 → 0.0");
    }

    #[test]
    fn rgba_alpha_midpoint_is_half() {
        let src = make_src(1, 1, 4, vec![0, 0, 0, 128]);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_oklab(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let alpha = dst.data.borrow()[3];
        let expected = 128.0 / 255.0;
        assert!(
            (alpha - expected).abs() < 1e-4,
            "alpha 128 → {expected}, got {alpha}"
        );
    }

    #[test]
    fn rgba_lab_channels_are_unaffected_by_alpha() {
        // The same RGB with different alpha values must produce identical L/a/b
        let src_opaque = make_src(1, 1, 4, vec![200, 100, 50, 255]);
        let src_transp = make_src(1, 1, 4, vec![200, 100, 50, 0]);
        let mut dst_opaque = make_dst(1, 1, 4);
        let mut dst_transp = make_dst(1, 1, 4);
        rgba_to_oklab(&src_opaque, &mut dst_opaque, TransferFunction::Srgb).unwrap();
        rgba_to_oklab(&src_transp, &mut dst_transp, TransferFunction::Srgb).unwrap();
        let a = dst_opaque.data.borrow();
        let b = dst_transp.data.borrow();
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-5,
                "channel {i}: alpha should not affect L/a/b ({} vs {})",
                a[i],
                b[i]
            );
        }
    }

    // ── BGR / BGRA channel ordering ───────────────────────────────────────────

    #[test]
    fn bgr_and_rgb_produce_same_oklab_for_grey() {
        // Grey is channel-order-invariant — a safe sanity check
        let src_rgb = make_src(1, 1, 3, vec![128, 128, 128]);
        let src_bgr = make_src(1, 1, 3, vec![128, 128, 128]);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        rgb_to_oklab(&src_rgb, &mut dst_rgb, TransferFunction::Srgb).unwrap();
        bgr_to_oklab(&src_bgr, &mut dst_bgr, TransferFunction::Srgb).unwrap();
        let r = dst_rgb.data.borrow();
        let b = dst_bgr.data.borrow();
        for i in 0..3 {
            assert!(
                (r[i] - b[i]).abs() < 1e-5,
                "channel {i}: grey should give identical oklab for rgb vs bgr"
            );
        }
    }

    #[test]
    fn bgr_and_rgb_differ_for_non_grey_input() {
        let pixel = vec![200u8, 100, 50]; // R≠G≠B
        let src_rgb = make_src(1, 1, 3, pixel.clone());
        let src_bgr = make_src(1, 1, 3, pixel); // same bytes, interpreted as B/G/R
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        rgb_to_oklab(&src_rgb, &mut dst_rgb, TransferFunction::Srgb).unwrap();
        bgr_to_oklab(&src_bgr, &mut dst_bgr, TransferFunction::Srgb).unwrap();
        let r = dst_rgb.data.borrow();
        let b = dst_bgr.data.borrow();
        assert!(
            (r[0] - b[0]).abs() > 0.01 || (r[1] - b[1]).abs() > 0.01,
            "rgb and bgr should differ for a non-grey pixel"
        );
    }

    // ── multi-pixel / stride ──────────────────────────────────────────────────

    #[test]
    fn multi_pixel_row_each_pixel_converted_independently() {
        // Two pixels: black then white
        let src = make_src(2, 1, 3, vec![0, 0, 0, 255, 255, 255]);
        let mut dst = make_dst(2, 1, 3);
        rgb_to_oklab(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d[0].abs() < 0.005, "pixel 0 L should be ≈ 0");
        assert!((d[3] - 1.0).abs() < 0.005, "pixel 1 L should be ≈ 1");
    }

    #[test]
    fn multi_row_image_respects_stride() {
        // 1×2 image: row 0 = black, row 1 = white
        let src = make_src(1, 2, 3, vec![0, 0, 0, 255, 255, 255]);
        let mut dst = make_dst(1, 2, 3);
        rgb_to_oklab(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d[0].abs() < 0.005, "row 0 L should be ≈ 0");
        assert!((d[3] - 1.0).abs() < 0.005, "row 1 L should be ≈ 1");
    }

    #[test]
    fn oklch_l_is_bounded_for_primary_colours() {
        for r in OKLAB_REFS {
            let src = make_src(1, 1, 3, r.rgb.to_vec());
            let mut dst = make_dst(1, 1, 3);
            rgb_to_oklch(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let l = dst.data.borrow()[0];
            assert!(
                l >= 0.0 && l <= 1.0,
                "rgb={:?}: Oklch L={l} is outside [0,1]",
                r.rgb
            );
        }
    }

    // ── transfer-function variant ─────────────────────────────────────────────

    #[test]
    fn linear_and_srgb_transfer_functions_differ_for_mid_grey() {
        let src_srgb = make_src(1, 1, 3, vec![128, 128, 128]);
        let src_linear = make_src(1, 1, 3, vec![128, 128, 128]);
        let mut dst_srgb = make_dst(1, 1, 3);
        let mut dst_linear = make_dst(1, 1, 3);
        rgb_to_oklab(&src_srgb, &mut dst_srgb, TransferFunction::Srgb).unwrap();
        rgb_to_oklab(&src_linear, &mut dst_linear, TransferFunction::Linear).unwrap();
        let ls = dst_srgb.data.borrow()[0];
        let ll = dst_linear.data.borrow()[0];
        assert!(
            (ls - ll).abs() > 0.01,
            "sRGB and Linear transfer functions should give different L for mid-grey \
             (sRGB L={ls}, linear L={ll})"
        );
    }
}
