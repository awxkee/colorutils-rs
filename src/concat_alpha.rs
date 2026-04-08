/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::{ColorError, ImageBuffer, ImageBufferMut};

/// Adds alpha plane into an existing RGB/XYZ/LAB or other 3 plane image. Image will become RGBA, XYZa, LABa etc.
pub fn append_alpha(
    src: &ImageBuffer<f32>,
    dst: &mut ImageBufferMut<f32>,
    a_plane: &ImageBuffer<f32>,
) -> Result<(), ColorError> {
    src.validate()?;
    dst.validate()?;
    a_plane.validate()?;
    if src.channels != 3 {
        return Err(ColorError::Generic(
            "We can add alpha only to 3 channels image".to_string(),
        ));
    }
    if dst.channels != 4 {
        return Err(ColorError::Generic(
            "Dst image should be 4 channels".to_string(),
        ));
    }
    if a_plane.channels != 1 {
        return Err(ColorError::Generic(
            "Alpha plane must have 1 channels".to_string(),
        ));
    }
    dst.try_match_immutable(src)?;
    dst.try_match_immutable(a_plane)?;

    let a_working_width = a_plane.width * a_plane.channels;
    let src_working_width = src.width * src.channels;
    let dst_working_width = dst.width * dst.channels;
    let dst_stride = dst.stride();
    for ((src, dst), a_plane) in src
        .data
        .chunks(src.stride())
        .zip(dst.data.borrow_mut().chunks_mut(dst_stride))
        .zip(a_plane.data.chunks(a_plane.stride()))
    {
        let src = &src[..src_working_width as usize];
        let dst = &mut dst[..dst_working_width as usize];
        let a_plane = &a_plane[..a_working_width as usize];
        for ((src, dst), a_ch) in src
            .as_chunks::<3>()
            .0
            .iter()
            .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            .zip(a_plane.iter())
        {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = *a_ch;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BufferStore;

    fn make_src(width: u32, height: u32) -> ImageBuffer<'static, f32> {
        let stride = width * 3;
        let data = (0..stride * height)
            .map(|i| (i as f32) / (stride * height) as f32)
            .collect::<Vec<_>>();
        ImageBuffer::from_vec(data, width, height, stride, 3).unwrap()
    }

    fn make_alpha(width: u32, height: u32, fill: f32) -> ImageBuffer<'static, f32> {
        let stride = width;
        let data = vec![fill; (stride * height) as usize];
        ImageBuffer::from_vec(data, width, height, stride, 1).unwrap()
    }

    fn make_dst(width: u32, height: u32) -> ImageBufferMut<'static, f32> {
        let stride = width * 4;
        let data = BufferStore::Owned(vec![0f32; (stride * height) as usize]);
        ImageBufferMut::new(data, width, height, stride, 4).unwrap()
    }

    #[test]
    fn appends_alpha_values_correctly() {
        let src = make_src(2, 2);
        let alpha = make_alpha(2, 2, 0.5);
        let mut dst = make_dst(2, 2);

        append_alpha(&src, &mut dst, &alpha).unwrap();

        let out = dst.data.borrow();
        // every fourth element (the alpha channel) must equal 0.5
        for pixel in out.chunks(4) {
            assert_eq!(pixel[3], 0.5, "alpha channel mismatch");
        }
    }

    #[test]
    fn rgb_channels_are_preserved() {
        let src = make_src(2, 1);
        let alpha = make_alpha(2, 1, 1.0);
        let mut dst = make_dst(2, 1);

        append_alpha(&src, &mut dst, &alpha).unwrap();

        let src_data = src.data;
        let out = dst.data.borrow();
        // pixels 0 and 1 — R, G, B must match src exactly
        assert_eq!(out[0], src_data[0]);
        assert_eq!(out[1], src_data[1]);
        assert_eq!(out[2], src_data[2]);
        assert_eq!(out[4], src_data[3]);
        assert_eq!(out[5], src_data[4]);
        assert_eq!(out[6], src_data[5]);
    }

    #[test]
    fn works_with_single_pixel() {
        let src = ImageBuffer::new(&[0.1f32, 0.2, 0.3], 1, 1, 3, 3).unwrap();
        let alpha = ImageBuffer::new(&[0.9f32], 1, 1, 1, 1).unwrap();
        let mut dst = make_dst(1, 1);

        append_alpha(&src, &mut dst, &alpha).unwrap();

        let out = dst.data.borrow();
        assert_eq!(out[0], 0.1);
        assert_eq!(out[1], 0.2);
        assert_eq!(out[2], 0.3);
        assert_eq!(out[3], 0.9);
    }

    // --- channel count validation ---

    #[test]
    fn rejects_src_with_wrong_channel_count() {
        let stride = 4;
        let data = vec![0f32; (stride * 1) as usize];
        let src = ImageBuffer::new(data.leak(), 1, 1, stride, 4).unwrap();
        let alpha = make_alpha(1, 1, 1.0);
        let mut dst = make_dst(1, 1);

        let err = append_alpha(&src, &mut dst, &alpha).unwrap_err();
        assert!(matches!(err, ColorError::Generic(_)));
    }

    #[test]
    fn rejects_dst_with_wrong_channel_count() {
        let src = make_src(1, 1);
        let alpha = make_alpha(1, 1, 1.0);
        let stride = 3;
        let data = BufferStore::Owned(vec![0f32; (stride * 1) as usize]);
        let mut dst = ImageBufferMut::new(data, 1, 1, stride, 3).unwrap();

        let err = append_alpha(&src, &mut dst, &alpha).unwrap_err();
        assert!(matches!(err, ColorError::Generic(_)));
    }

    #[test]
    fn rejects_alpha_plane_with_wrong_channel_count() {
        let src = make_src(1, 1);
        let stride = 2;
        let data = vec![0f32; (stride * 1) as usize];
        let alpha = ImageBuffer::new(data.leak(), 1, 1, stride, 2).unwrap();
        let mut dst = make_dst(1, 1);

        let err = append_alpha(&src, &mut dst, &alpha).unwrap_err();
        assert!(matches!(err, ColorError::Generic(_)));
    }

    #[test]
    fn rejects_mismatched_src_and_dst_dimensions() {
        let src = make_src(2, 2);
        let alpha = make_alpha(2, 2, 1.0);
        let mut dst = make_dst(4, 4);

        let err = append_alpha(&src, &mut dst, &alpha).unwrap_err();
        assert!(matches!(err, ColorError::DimensionMismatch { .. }));
    }

    #[test]
    fn rejects_mismatched_alpha_and_dst_dimensions() {
        let src = make_src(2, 2);
        let alpha = make_alpha(4, 4, 1.0);
        let mut dst = make_dst(2, 2);

        let err = append_alpha(&src, &mut dst, &alpha).unwrap_err();
        assert!(matches!(err, ColorError::DimensionMismatch { .. }));
    }
}
