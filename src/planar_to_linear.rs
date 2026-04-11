/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::{ColorError, ImageBuffer, ImageBufferMut, TransferFunction};

fn channels_to_linear(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    dst.validate()?;
    src.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 1 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }

    let mut lut_table = [0f32; 256];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = transfer_function.linearize(i as f32 * (1. / 255.0));
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
            *dst = lut_table[src as usize];
        }
    }

    Ok(())
}

/// This function converts Plane to Linear. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn plane_to_linear(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_linear(src, dst, transfer_function)
}

#[cfg(test)]
mod tests_plane_to_linear {
    use super::*;
    use crate::BufferStore;

    fn make_src(data: Vec<u8>, width: u32, height: u32) -> ImageBuffer<'static, u8> {
        ImageBuffer::from_vec(data, width, height, 1).unwrap()
    }

    fn make_dst(width: u32, height: u32) -> ImageBufferMut<'static, f32> {
        ImageBufferMut::new(
            BufferStore::Owned(vec![0f32; (width * height) as usize]),
            width,
            height,
            1,
        )
        .unwrap()
    }

    fn assert_approx(a: f32, b: f32, eps: f32, label: &str) {
        assert!((a - b).abs() < eps, "{label}: got {a}, expected {b}");
    }

    // ── boundary values ───────────────────────────────────────────────────────

    #[test]
    fn zero_maps_to_zero() {
        let src = make_src(vec![0], 1, 1);
        let mut dst = make_dst(1, 1);
        plane_to_linear(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_eq!(dst.data.borrow()[0], 0.0);
    }

    #[test]
    fn max_maps_to_one() {
        let src = make_src(vec![255], 1, 1);
        let mut dst = make_dst(1, 1);
        plane_to_linear(&src, &mut dst, TransferFunction::Srgb).unwrap();
        assert_approx(dst.data.borrow()[0], 1.0, 1e-5, "255 → 1.0");
    }

    // ── lut correctness ───────────────────────────────────────────────────────

    #[test]
    fn lut_matches_direct_linearize() {
        for input in 0u8..=255 {
            let expected = TransferFunction::Srgb.linearize(input as f32 / 255.0);
            let src = make_src(vec![input], 1, 1);
            let mut dst = make_dst(1, 1);
            plane_to_linear(&src, &mut dst, TransferFunction::Srgb).unwrap();
            assert_approx(
                dst.data.borrow()[0],
                expected,
                1e-6,
                &format!("input={input}"),
            );
        }
    }

    #[test]
    fn lut_matches_direct_linearize_linear_transfer() {
        for input in (0u8..=255).step_by(16) {
            let expected = TransferFunction::Linear.linearize(input as f32 / 255.0);
            let src = make_src(vec![input], 1, 1);
            let mut dst = make_dst(1, 1);
            plane_to_linear(&src, &mut dst, TransferFunction::Linear).unwrap();
            assert_approx(
                dst.data.borrow()[0],
                expected,
                1e-6,
                &format!("input={input}"),
            );
        }
    }

    // ── linear transfer is identity ───────────────────────────────────────────

    #[test]
    fn linear_transfer_is_identity() {
        for input in (0u8..=255).step_by(16) {
            let src = make_src(vec![input], 1, 1);
            let mut dst = make_dst(1, 1);
            plane_to_linear(&src, &mut dst, TransferFunction::Linear).unwrap();
            let expected = input as f32 / 255.0;
            assert_approx(
                dst.data.borrow()[0],
                expected,
                1e-6,
                &format!("input={input}"),
            );
        }
    }

    // ── output range ──────────────────────────────────────────────────────────

    #[test]
    fn output_always_in_0_1_range() {
        let data: Vec<u8> = (0u8..=255).collect();
        let src = make_src(data, 256, 1);
        let mut dst = make_dst(256, 1);
        plane_to_linear(&src, &mut dst, TransferFunction::Srgb).unwrap();
        for (i, &v) in dst.data.borrow().iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "output out of range at index {i}: {v}"
            );
        }
    }

    // ── monotonicity ──────────────────────────────────────────────────────────

    #[test]
    fn output_monotonically_non_decreasing() {
        let data: Vec<u8> = (0u8..=255).collect();
        let src = make_src(data, 256, 1);
        let mut dst = make_dst(256, 1);
        plane_to_linear(&src, &mut dst, TransferFunction::Srgb).unwrap();
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

    // ── roundtrip ─────────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_srgb() {
        for original in 0u8..=255 {
            let src = make_src(vec![original], 1, 1);
            let mut dst = make_dst(1, 1);
            plane_to_linear(&src, &mut dst, TransferFunction::Srgb).unwrap();
            let linear = dst.data.borrow()[0];
            // gamma(linear) * 255 should recover original within 1 DN
            let recovered = (TransferFunction::Srgb.gamma(linear) * 255.).round() as u8;
            assert!(
                (recovered as i16 - original as i16).abs() <= 1,
                "original={original} recovered={recovered}"
            );
        }
    }

    // ── multi-pixel / multi-row consistency ───────────────────────────────────

    #[test]
    fn multi_pixel_consistent() {
        let src = make_src(vec![128u8; 8], 8, 1);
        let mut dst = make_dst(8, 1);
        plane_to_linear(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d.iter().all(|&v| v == d[0]), "inconsistent pixels: {d:?}");
    }

    #[test]
    fn multi_row_consistent() {
        let src = make_src(vec![200u8; 4], 2, 2);
        let mut dst = make_dst(2, 2);
        plane_to_linear(&src, &mut dst, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d.iter().all(|&v| v == d[0]), "row mismatch: {d:?}");
    }

    // ── error paths ───────────────────────────────────────────────────────────

    #[test]
    fn rejects_multi_channel_src() {
        let src = ImageBuffer::from_vec(vec![0u8; 3], 1, 1, 3).unwrap();
        let mut dst = make_dst(1, 1);
        assert!(plane_to_linear(&src, &mut dst, TransferFunction::Srgb).is_err());
    }
}
