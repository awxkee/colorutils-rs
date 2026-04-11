/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::xyz_target::XyzTarget;
use crate::{ColorError, ImageBuffer, ImageBufferMut, LCh, Lab, Luv, Rgb, TransferFunction, Xyz};

fn channels_to_xyz_with_alpha<const CHANNELS_CONFIGURATION: u8, const TARGET: u8>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    src.validate()?;
    dst.validate()?;
    dst.try_match_immutable_with_channels(src)?;
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let mut _wide_row_handler: Option<
        unsafe fn(usize, &[f32], &mut [f32], u32, &[[f32; 3]; 3]) -> usize,
    > = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            use crate::sse::sse_channels_to_xyza_laba;
            _wide_row_handler = Some(sse_channels_to_xyza_laba::<CHANNELS_CONFIGURATION, TARGET>);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::neon::neon_channels_to_xyza_or_laba;
        _wide_row_handler = Some(neon_channels_to_xyza_or_laba::<CHANNELS_CONFIGURATION, TARGET>);
    }

    let channels = image_configuration.channel_count();

    let mut lut_table = [0.; 256];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = transfer_function.linearize(i as f32 * (1. / 255.0));
    }

    let mut a_lut_table = [0.; 256];
    for (i, lut) in a_lut_table.iter_mut().enumerate() {
        *lut = i as f32 * (1. / 255.0);
    }

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;
    let mut transient_row = vec![0f32; src.width as usize * channels];
    let width = src.width;

    for (dst, src) in dst
        .data
        .borrow_mut()
        .chunks_mut(dst_stride)
        .zip(src.data.chunks(src.stride()))
    {
        let src = &src[..src_r_width as usize];
        let dst = &mut dst[..dst_r_width as usize];

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            for (dst, src) in transient_row
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                dst[0] = lut_table[src[image_configuration.r_index()] as usize];
                dst[1] = lut_table[src[image_configuration.g_index()] as usize];
                dst[2] = lut_table[src[image_configuration.b_index()] as usize];
            }
        } else if image_configuration == ImageConfiguration::Bgra
            || image_configuration == ImageConfiguration::Rgba
        {
            for (dst, src) in transient_row
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                dst[0] = lut_table[src[image_configuration.r_index()] as usize];
                dst[1] = lut_table[src[image_configuration.g_index()] as usize];
                dst[2] = lut_table[src[image_configuration.b_index()] as usize];
                dst[3] = a_lut_table[src[image_configuration.a_index()] as usize];
            }
        }

        let mut cx = 0usize;

        if let Some(dispatcher) = _wide_row_handler {
            cx = unsafe { dispatcher(cx, &transient_row, dst, width, matrix) };
        }

        if image_configuration == ImageConfiguration::Bgr
            || image_configuration == ImageConfiguration::Rgb
        {
            let src = &transient_row[cx * 3..];
            let dst = &mut dst[cx * 3..];
            for (dst, src) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0.iter())
            {
                let rgb = Rgb::<f32>::new(src[0], src[1], src[2]);
                let xyz = Xyz::from_linear_rgb(rgb, matrix);
                match target {
                    XyzTarget::Lab => {
                        let lab = Lab::from_xyz(xyz);
                        dst[0] = lab.l;
                        dst[1] = lab.a;
                        dst[2] = lab.b;
                    }
                    XyzTarget::Xyz => {
                        dst[0] = xyz.x;
                        dst[1] = xyz.y;
                        dst[2] = xyz.z;
                    }
                    XyzTarget::Luv => {
                        let luv = Luv::from_xyz(xyz);
                        dst[0] = luv.l;
                        dst[1] = luv.u;
                        dst[2] = luv.v;
                    }
                    XyzTarget::Lch => {
                        let luv = Luv::from_xyz(xyz);
                        let lch = LCh::from_luv(luv);
                        dst[0] = lch.l;
                        dst[1] = lch.c;
                        dst[2] = lch.h;
                    }
                }
            }
        } else if image_configuration == ImageConfiguration::Bgra
            || image_configuration == ImageConfiguration::Rgba
        {
            let src = &transient_row[cx * 4..];
            let dst = &mut dst[cx * 4..];
            for (dst, src) in dst
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<4>().0.iter())
            {
                let rgb = Rgb::<f32>::new(src[0], src[1], src[2]);
                let xyz = Xyz::from_linear_rgb(rgb, matrix);
                match target {
                    XyzTarget::Lab => {
                        let lab = Lab::from_xyz(xyz);
                        dst[0] = lab.l;
                        dst[1] = lab.a;
                        dst[2] = lab.b;
                    }
                    XyzTarget::Xyz => {
                        dst[0] = xyz.x;
                        dst[1] = xyz.y;
                        dst[2] = xyz.z;
                    }
                    XyzTarget::Luv => {
                        let luv = Luv::from_xyz(xyz);
                        dst[0] = luv.l;
                        dst[1] = luv.u;
                        dst[2] = luv.v;
                    }
                    XyzTarget::Lch => {
                        let luv = Luv::from_xyz(xyz);
                        let lch = LCh::from_luv(luv);
                        dst[0] = lch.l;
                        dst[1] = lch.c;
                        dst[2] = lch.h;
                    }
                }
                dst[3] = src[3];
            }
        }
    }

    Ok(())
}

/// This function converts RGBA to CIE L*ab.
///
/// This function converts RGBA to CIE L*ab against D65 white point and preserving
/// and normalizing alpha channels keeping it at last positions.
/// This is much more effective than naive direct transformation
///
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn rgba_to_lab_with_alpha(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

/// This function converts BGRA to CIE L*ab.
///
/// This function converts BGRA to CIE L*ab against D65 white point
/// and preserving and normalizing alpha channels keeping it at last positions.
/// This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn bgra_to_lab_with_alpha(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

/// This function converts RGBA to CIE L*uv.
///
/// This function converts RGBA to CIE L*uv against D65 white point and preserving
/// and normalizing alpha channels keeping it at last positions.
/// This is much more effective than naive direct transformation.
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn rgba_to_luv_with_alpha(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

/// This function converts BGRA to CIE L*uv.
///
/// This function converts BGRA to CIE L*uv against D65 white point
/// and preserving and normalizing alpha channels keeping it at last positions.
/// This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn bgra_to_luv_with_alpha(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

/// This function converts RGBA to CIE XYZ against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn rgba_to_xyz_with_alpha(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

/// This function converts BGRA to CIE XYZ against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn bgra_to_xyz_with_alpha(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

/// This function converts RGBA to CIE LCH against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH(a) data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn rgba_to_lch_with_alpha(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Rgba as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

/// This function converts BGRA to CIE LCH against D65 white point and preserving and normalizing alpha channels keeping it at last positions. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LCH data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn bgra_to_lch_with_alpha(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz_with_alpha::<{ ImageConfiguration::Bgra as u8 }, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        matrix,
        transfer_function,
    )
}

#[cfg(test)]
mod tests_with_alpha {
    use super::*;
    use crate::{BufferStore, SRGB_TO_XYZ_D65};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_src(data: Vec<u8>, width: u32, height: u32, channels: u32) -> ImageBuffer<'static, u8> {
        ImageBuffer::from_vec(data, width, height, channels).unwrap()
    }

    fn make_dst(width: u32, height: u32, channels: u32) -> ImageBufferMut<'static, f32> {
        ImageBufferMut::new(
            BufferStore::Owned(vec![0f32; (width * height * channels) as usize]),
            width,
            height,
            channels,
        )
        .unwrap()
    }

    fn assert_approx(a: f32, b: f32, eps: f32, label: &str) {
        assert!((a - b).abs() < eps, "{label}: expected {b:.6}, got {a:.6}");
    }

    // dst layout: [X/L, Y/a/u/C, Z/b/v/h, A] per pixel
    fn read_pixel(dst: &ImageBufferMut<'_, f32>, px: usize) -> [f32; 4] {
        let d = dst.data.borrow();
        [d[px * 4], d[px * 4 + 1], d[px * 4 + 2], d[px * 4 + 3]]
    }

    // ── rgba_to_xyz_with_alpha ────────────────────────────────────────────────

    #[test]
    fn rgba_to_xyz_with_alpha_black_opaque() {
        let src = make_src(vec![0, 0, 0, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_xyz_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p = read_pixel(&dst, 0);
        assert_approx(p[0], 0.0, 1e-5, "X");
        assert_approx(p[1], 0.0, 1e-5, "Y");
        assert_approx(p[2], 0.0, 1e-5, "Z");
        assert_approx(p[3], 1.0, 1e-5, "A");
    }

    #[test]
    fn rgba_to_xyz_with_alpha_white_opaque() {
        let src = make_src(vec![255, 255, 255, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_xyz_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p = read_pixel(&dst, 0);
        assert_approx(p[0], 0.9505, 1e-3, "X");
        assert_approx(p[1], 1.0000, 1e-3, "Y");
        assert_approx(p[2], 1.0890, 1e-3, "Z");
        assert_approx(p[3], 1.0, 1e-5, "A");
    }

    #[test]
    fn rgba_to_xyz_with_alpha_transparent() {
        let src = make_src(vec![255, 255, 255, 0], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_xyz_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p = read_pixel(&dst, 0);
        assert_approx(p[3], 0.0, 1e-5, "A fully transparent");
    }

    #[test]
    fn rgba_to_xyz_with_alpha_half_alpha() {
        let src = make_src(vec![255, 255, 255, 128], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_xyz_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p = read_pixel(&dst, 0);
        assert_approx(p[3], 128.0 / 255.0, 1e-4, "A half");
    }

    #[test]
    fn rgba_to_xyz_with_alpha_xyz_unaffected_by_alpha() {
        // Same RGB, different alpha → XYZ channels must be identical
        let src_full = make_src(vec![100, 149, 237, 255], 1, 1, 4);
        let src_half = make_src(vec![100, 149, 237, 128], 1, 1, 4);
        let mut dst_full = make_dst(1, 1, 4);
        let mut dst_half = make_dst(1, 1, 4);
        rgba_to_xyz_with_alpha(
            &src_full,
            &mut dst_full,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        rgba_to_xyz_with_alpha(
            &src_half,
            &mut dst_half,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        let pf = read_pixel(&dst_full, 0);
        let ph = read_pixel(&dst_half, 0);
        for ch in 0..3 {
            assert_approx(
                pf[ch],
                ph[ch],
                1e-5,
                &format!("XYZ ch{ch} alpha-independent"),
            );
        }
    }

    // ── bgra_to_xyz_with_alpha ────────────────────────────────────────────────

    #[test]
    fn bgra_to_xyz_with_alpha_white() {
        let src = make_src(vec![255, 255, 255, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        bgra_to_xyz_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p = read_pixel(&dst, 0);
        assert_approx(p[0], 0.9505, 1e-3, "X");
        assert_approx(p[1], 1.0000, 1e-3, "Y");
        assert_approx(p[2], 1.0890, 1e-3, "Z");
        assert_approx(p[3], 1.0, 1e-5, "A");
    }

    #[test]
    fn bgra_to_xyz_with_alpha_grey_matches_rgba() {
        // Grey is BGR-invariant
        let data = vec![128u8, 128, 128, 200];
        let src_rgba = make_src(data.clone(), 1, 1, 4);
        let src_bgra = make_src(data, 1, 1, 4);
        let mut dst_rgba = make_dst(1, 1, 4);
        let mut dst_bgra = make_dst(1, 1, 4);
        rgba_to_xyz_with_alpha(
            &src_rgba,
            &mut dst_rgba,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgra_to_xyz_with_alpha(
            &src_bgra,
            &mut dst_bgra,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        for ch in 0..4 {
            assert_approx(
                read_pixel(&dst_rgba, 0)[ch],
                read_pixel(&dst_bgra, 0)[ch],
                1e-5,
                &format!("ch{ch}"),
            );
        }
    }

    // ── rgba_to_lab_with_alpha ────────────────────────────────────────────────

    #[test]
    fn rgba_to_lab_with_alpha_white() {
        let src = make_src(vec![255, 255, 255, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_lab_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p = read_pixel(&dst, 0);
        assert_approx(p[0], 100.0, 0.5, "L*");
        assert_approx(p[1], 0.0, 0.5, "a*");
        assert_approx(p[2], 0.0, 0.5, "b*");
        assert_approx(p[3], 1.0, 1e-5, "A");
    }

    #[test]
    fn rgba_to_lab_with_alpha_black() {
        let src = make_src(vec![0, 0, 0, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        rgba_to_lab_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p = read_pixel(&dst, 0);
        assert_approx(p[0], 0.0, 0.1, "L* black");
        assert_approx(p[3], 1.0, 1e-5, "A");
    }

    #[test]
    fn rgba_to_lab_with_alpha_alpha_passthrough() {
        for alpha in [0u8, 64, 128, 192, 255] {
            let src = make_src(vec![200, 100, 50, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            rgba_to_lab_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb)
                .unwrap();
            assert_approx(
                read_pixel(&dst, 0)[3],
                alpha as f32 / 255.0,
                1e-4,
                &format!("alpha={alpha}"),
            );
        }
    }

    // ── bgra_to_lab_with_alpha ────────────────────────────────────────────────

    #[test]
    fn bgra_to_lab_with_alpha_white() {
        let src = make_src(vec![255, 255, 255, 255], 1, 1, 4);
        let mut dst = make_dst(1, 1, 4);
        bgra_to_lab_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p = read_pixel(&dst, 0);
        assert_approx(p[0], 100.0, 0.5, "L*");
        assert_approx(p[3], 1.0, 1e-5, "A");
    }

    #[test]
    fn bgra_to_lab_with_alpha_grey_matches_rgba() {
        let data = vec![150u8, 150, 150, 180];
        let src_rgba = make_src(data.clone(), 1, 1, 4);
        let src_bgra = make_src(data, 1, 1, 4);
        let mut dst_rgba = make_dst(1, 1, 4);
        let mut dst_bgra = make_dst(1, 1, 4);
        rgba_to_lab_with_alpha(
            &src_rgba,
            &mut dst_rgba,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgra_to_lab_with_alpha(
            &src_bgra,
            &mut dst_bgra,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        for ch in 0..4 {
            assert_approx(
                read_pixel(&dst_rgba, 0)[ch],
                read_pixel(&dst_bgra, 0)[ch],
                1e-4,
                &format!("ch{ch}"),
            );
        }
    }

    #[test]
    fn rgba_to_luv_with_alpha_alpha_passthrough() {
        for alpha in [0u8, 128, 255] {
            let src = make_src(vec![100, 100, 100, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            rgba_to_luv_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb)
                .unwrap();
            assert_approx(
                read_pixel(&dst, 0)[3],
                alpha as f32 / 255.0,
                1e-4,
                &format!("alpha={alpha}"),
            );
        }
    }

    #[test]
    fn bgra_to_luv_with_alpha_grey_matches_rgba() {
        let data = vec![180u8, 180, 180, 100];
        let src_rgba = make_src(data.clone(), 1, 1, 4);
        let src_bgra = make_src(data, 1, 1, 4);
        let mut dst_rgba = make_dst(1, 1, 4);
        let mut dst_bgra = make_dst(1, 1, 4);
        rgba_to_luv_with_alpha(
            &src_rgba,
            &mut dst_rgba,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgra_to_luv_with_alpha(
            &src_bgra,
            &mut dst_bgra,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        for ch in 0..4 {
            assert_approx(
                read_pixel(&dst_rgba, 0)[ch],
                read_pixel(&dst_bgra, 0)[ch],
                1e-4,
                &format!("ch{ch}"),
            );
        }
    }

    #[test]
    fn rgba_to_lch_with_alpha_alpha_passthrough() {
        for alpha in [0u8, 128, 255] {
            let src = make_src(vec![200, 200, 200, alpha], 1, 1, 4);
            let mut dst = make_dst(1, 1, 4);
            rgba_to_lch_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb)
                .unwrap();
            assert_approx(
                read_pixel(&dst, 0)[3],
                alpha as f32 / 255.0,
                1e-4,
                &format!("alpha={alpha}"),
            );
        }
    }

    #[test]
    fn bgra_to_lch_with_alpha_grey_matches_rgba() {
        let data = vec![90u8, 90, 90, 220];
        let src_rgba = make_src(data.clone(), 1, 1, 4);
        let src_bgra = make_src(data, 1, 1, 4);
        let mut dst_rgba = make_dst(1, 1, 4);
        let mut dst_bgra = make_dst(1, 1, 4);
        rgba_to_lch_with_alpha(
            &src_rgba,
            &mut dst_rgba,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgra_to_lch_with_alpha(
            &src_bgra,
            &mut dst_bgra,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        for ch in 0..4 {
            assert_approx(
                read_pixel(&dst_rgba, 0)[ch],
                read_pixel(&dst_bgra, 0)[ch],
                1e-4,
                &format!("ch{ch}"),
            );
        }
    }

    // ── multi-pixel consistency ───────────────────────────────────────────────

    #[test]
    fn rgba_to_lab_with_alpha_multi_pixel_consistent() {
        let pixel = [100u8, 149, 237, 200];
        let src = make_src(pixel.repeat(4), 4, 1, 4);
        let mut dst = make_dst(4, 1, 4);
        rgba_to_lab_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p0 = read_pixel(&dst, 0);
        for px in 1..4 {
            let p = read_pixel(&dst, px);
            for ch in 0..4 {
                assert_approx(p[ch], p0[ch], 1e-4, &format!("px{px} ch{ch}"));
            }
        }
    }

    #[test]
    fn rgba_to_xyz_with_alpha_multi_row_consistent() {
        let src = make_src(vec![128u8, 64, 32, 255].repeat(2), 1, 2, 4);
        let mut dst = make_dst(1, 2, 4);
        rgba_to_xyz_with_alpha(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let p0 = read_pixel(&dst, 0);
        let p1 = read_pixel(&dst, 1);
        for ch in 0..4 {
            assert_approx(p0[ch], p1[ch], 1e-5, &format!("row mismatch ch{ch}"));
        }
    }
}
