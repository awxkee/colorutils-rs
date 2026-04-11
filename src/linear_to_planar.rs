/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::{ColorError, ImageBuffer, ImageBufferMut, TransferFunction};

#[allow(clippy::type_complexity)]
fn linear_to_gamma_channels(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    dst.validate()?;
    src.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 1 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }

    let mut lut_table = [0u8; 65535];
    for (i, lut) in lut_table[..2049].iter_mut().enumerate() {
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
        for (dst, &src) in dst.iter_mut().zip(src.iter()) {
            let px = (src * 2048f32).min(2048.).round() as usize;
            *dst = lut_table[px];
        }
    }

    Ok(())
}

/// This function converts Linear to Plane. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear plane data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive Gamma plane data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_to_plane(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    linear_to_gamma_channels(src, dst, transfer_function)
}

#[cfg(test)]
mod tests_linear_to_gamma {
    use super::*;
    use crate::BufferStore;

    fn make_src(data: Vec<f32>, width: u32, height: u32) -> ImageBuffer<'static, f32> {
        ImageBuffer::from_vec(data, width, height, 1).unwrap()
    }

    fn make_dst(width: u32, height: u32) -> ImageBufferMut<'static, u8> {
        ImageBufferMut::new(
            BufferStore::Owned(vec![0u8; (width * height) as usize]),
            width,
            height,
            1,
        )
        .unwrap()
    }

    // ── boundary values ───────────────────────────────────────────────────────

    #[test]
    fn zero_maps_to_zero() {
        let src = make_src(vec![0.0], 1, 1);
        let mut dst = make_dst(1, 1);
        linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data.borrow()[0], 0);
    }

    #[test]
    fn one_maps_to_255() {
        let src = make_src(vec![1.0], 1, 1);
        let mut dst = make_dst(1, 1);
        linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data.borrow()[0], 255);
    }

    #[test]
    fn above_one_clamps_to_255() {
        for val in [1.001f32, 2.0, 100.0, f32::MAX] {
            let src = make_src(vec![val], 1, 1);
            let mut dst = make_dst(1, 1);
            linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
            assert_eq!(dst.data.borrow()[0], 255, "val={val}");
        }
    }

    #[test]
    fn negative_clamps_to_zero() {
        // negative * 2048 rounds to 0 index → lut[0] = 0
        let src = make_src(vec![-0.5], 1, 1);
        let mut dst = make_dst(1, 1);
        linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data.borrow()[0], 0);
    }

    // ── roundtrip: gamma → linear → gamma should recover original ────────────

    #[test]
    fn roundtrip_srgb() {
        for original in (0u8..=255).step_by(8) {
            let linear = TransferFunction::Srgb.linearize(original as f32 / 255.0);
            let src = make_src(vec![linear], 1, 1);
            let mut dst = make_dst(1, 1);
            linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let recovered = dst.data.borrow()[0];
            // LUT has 2048 steps → max rounding error is ~1 DN
            assert!(
                (recovered as i16 - original as i16).abs() <= 2,
                "original={original} recovered={recovered}"
            );
        }
    }

    #[test]
    fn roundtrip_linear_transfer() {
        for original in (0u8..=255).step_by(16) {
            let linear = TransferFunction::Linear.linearize(original as f32 / 255.0);
            let src = make_src(vec![linear], 1, 1);
            let mut dst = make_dst(1, 1);
            linear_to_plane(&src, &mut dst, TransferFunction::Linear).unwrap();
            let recovered = dst.data.borrow()[0];
            assert!(
                (recovered as i16 - original as i16).abs() <= 2,
                "original={original} recovered={recovered}"
            );
        }
    }

    // ── monotonicity ──────────────────────────────────────────────────────────

    #[test]
    fn output_is_monotonically_non_decreasing() {
        let inputs: Vec<f32> = (0..=64).map(|i| i as f32 / 64.0).collect();
        let src = make_src(inputs, 65, 1);
        let mut dst = make_dst(65, 1);
        linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        for i in 1..d.len() {
            assert!(
                d[i] >= d[i - 1],
                "non-monotone at index {i}: {} < {}",
                d[i],
                d[i - 1]
            );
        }
    }

    // ── multi-pixel / multi-row consistency ───────────────────────────────────

    #[test]
    fn multi_pixel_consistent() {
        let val = 0.5f32;
        let src = make_src(vec![val; 8], 8, 1);
        let mut dst = make_dst(8, 1);
        linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d.iter().all(|&v| v == d[0]), "inconsistent pixels: {d:?}");
    }

    #[test]
    fn multi_row_consistent() {
        let val = 0.3f32;
        let src = make_src(vec![val; 4], 2, 2);
        let mut dst = make_dst(2, 2);
        linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d.iter().all(|&v| v == d[0]), "row mismatch: {d:?}");
    }

    // ── wrong channel count returns error ─────────────────────────────────────

    #[test]
    fn rejects_multi_channel_src() {
        // channels=3 is not 1 → must error
        let src = ImageBuffer::from_vec(vec![0f32; 3], 1, 1, 3).unwrap();
        let mut dst = make_dst(1, 1);
        let result = linear_to_plane(&src, &mut dst, TransferFunction::Srgb);
        assert!(result.is_err());
    }

    // ── known values ─────────────────────────────────────────────────────────

    #[test]
    fn known_srgb_midtone() {
        // sRGB linearised 0.5 ≈ 0.2140, gamma applied back ≈ 128
        let linear = TransferFunction::Srgb.linearize(128.0 / 255.0);
        let src = make_src(vec![linear], 1, 1);
        let mut dst = make_dst(1, 1);
        linear_to_plane(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let v = dst.data.borrow()[0];
        assert!((v as i16 - 128).abs() <= 2, "midtone={v}");
    }

    #[test]
    fn linear_transfer_is_identity() {
        // For a linear transfer function gamma(x) == x, so output ≈ round(x * 255)
        let input = 0.6f32;
        let src = make_src(vec![input], 1, 1);
        let mut dst = make_dst(1, 1);
        linear_to_plane(&src, &mut dst, TransferFunction::Linear).unwrap();
        let expected = (input * 255.0).round() as u8;
        let got = dst.data.borrow()[0];
        assert!(
            (got as i16 - expected as i16).abs() <= 2,
            "expected≈{expected} got={got}"
        );
    }
}
