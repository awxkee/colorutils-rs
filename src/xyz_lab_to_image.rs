/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::xyz_target::XyzTarget;
use crate::{ColorError, ImageBuffer, ImageBufferMut, LCh, Lab, Luv, Rgb, XYZ_TO_SRGB_D65, Xyz};

pub(crate) fn xyz_target_to_rgb(source: XyzTarget, px: &[f32], matrix: &[[f32; 3]; 3]) -> Rgb<f32> {
    match source {
        XyzTarget::Lab => Lab::new(px[0], px[1], px[2]).to_linear_rgb(matrix),
        XyzTarget::Xyz => Xyz::new(px[0], px[1], px[2]).to_linear_rgb(matrix),
        XyzTarget::Luv => Luv::new(px[0], px[1], px[2]).to_linear_rgb(matrix),
        XyzTarget::Lch => LCh::new(px[0], px[1], px[2]).to_linear_rgb(matrix),
    }
}

fn write_lut_rgb(dst: &mut [u8], transient: &[f32], lut: &[u8; 2049], cfg: ImageConfiguration) {
    for (dst, src) in dst
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(transient.as_chunks::<3>().0.iter())
    {
        let r = (src[cfg.r_index()].clamp(0., 1.) * 2048.).round() as u16;
        let g = (src[cfg.g_index()].clamp(0., 1.) * 2048.).round() as u16;
        let b = (src[cfg.b_index()].clamp(0., 1.) * 2048.).round() as u16;
        dst[cfg.r_index()] = lut[r.min(2048) as usize];
        dst[cfg.g_index()] = lut[g.min(2048) as usize];
        dst[cfg.b_index()] = lut[b.min(2048) as usize];
    }
}

fn write_lut_rgba(dst: &mut [u8], transient: &[f32], lut: &[u8; 2049], cfg: ImageConfiguration) {
    for (dst, src) in dst
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(transient.as_chunks::<4>().0.iter())
    {
        let r = (src[cfg.r_index()].clamp(0., 1.) * 2048.).round() as u16;
        let g = (src[cfg.g_index()].clamp(0., 1.) * 2048.).round() as u16;
        let b = (src[cfg.b_index()].clamp(0., 1.) * 2048.).round() as u16;
        dst[cfg.r_index()] = lut[r.min(2048) as usize];
        dst[cfg.g_index()] = lut[g.min(2048) as usize];
        dst[cfg.b_index()] = lut[b.min(2048) as usize];
        dst[cfg.a_index()] = (src[cfg.a_index()] * 255.).clamp(0., 255.).round() as u8;
    }
}

fn xyz_to_channels<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool, const TARGET: u8>(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    a_plane: Option<&ImageBuffer<'_, f32>>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    dst.validate()?;
    src.validate()?;
    dst.try_match_immutable(src)?;
    if src.channels != 3 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    if USE_ALPHA && dst.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(4));
    } else if !USE_ALPHA && dst.channels != 3 {
        return Err(ColorError::UnsupportedChannelsCount(3));
    }
    if let Some(plane) = a_plane
        && USE_ALPHA
    {
        dst.try_match_immutable(plane)?;
    }

    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let source: XyzTarget = TARGET.into();

    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }
    if USE_ALPHA && a_plane.is_none() {
        panic!("Alpha plane must be provided when USE_ALPHA is true");
    }

    let mut _wide_row_handler: Option<
        unsafe fn(usize, &[f32], &[f32], &mut [f32], u32, &[[f32; 3]; 3]) -> usize,
    > = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        use crate::sse::sse_xyz_to_channels;
        _wide_row_handler = Some(sse_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        use crate::avx::avx_xyz_to_channels;
        _wide_row_handler = Some(avx_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::neon::neon_xyz_to_channels;
        _wide_row_handler = Some(neon_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    let channels = image_configuration.channel_count();

    let mut lut_table = [0u8; 2049];
    for (i, lut) in lut_table.iter_mut().enumerate() {
        *lut = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.)
            .round()
            .min(255.) as u8;
    }

    let dst_stride = dst.stride();
    let src_r_width = src.width * src.channels;
    let dst_r_width = dst.width * dst.channels;
    let width = src.width;

    let mut transient_row = vec![0f32; width as usize * channels];

    let src_rows = src.data.chunks(src.stride());
    let dst_rows = dst.data.borrow_mut().chunks_mut(dst_stride);

    if USE_ALPHA {
        let a_plane = a_plane.unwrap();
        for ((dst, src), a_row) in dst_rows
            .zip(src_rows)
            .zip(a_plane.data.chunks(a_plane.stride()))
        {
            let src = &src[..src_r_width as usize];
            let dst = &mut dst[..dst_r_width as usize];
            let a_row = &a_row[..width as usize];

            let mut cx = 0usize;
            if let Some(dispatcher) = _wide_row_handler {
                cx = unsafe { dispatcher(0, src, a_row, &mut transient_row, width, matrix) };
            }

            for ((transient, src), &a) in transient_row[cx * channels..]
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src[cx * 3..].as_chunks::<3>().0.iter())
                .zip(a_row[cx..].iter())
            {
                let rgb = xyz_target_to_rgb(source, src, matrix);
                transient[image_configuration.r_index()] = rgb.r;
                transient[image_configuration.g_index()] = rgb.g;
                transient[image_configuration.b_index()] = rgb.b;
                transient[image_configuration.a_index()] = a;
            }

            write_lut_rgba(dst, &transient_row, &lut_table, image_configuration);
        }
    } else {
        for (dst, src) in dst_rows.zip(src_rows) {
            let src = &src[..src_r_width as usize];
            let dst = &mut dst[..dst_r_width as usize];

            let mut cx = 0usize;
            if let Some(dispatcher) = _wide_row_handler {
                cx = unsafe { dispatcher(0, src, &[], &mut transient_row, width, matrix) };
            }

            for (transient, src) in transient_row[cx * channels..]
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src[cx * 3..].as_chunks::<3>().0.iter())
            {
                let rgb = xyz_target_to_rgb(source, src, matrix);
                transient[image_configuration.r_index()] = rgb.r;
                transient[image_configuration.g_index()] = rgb.g;
                transient[image_configuration.b_index()] = rgb.b;
            }

            write_lut_rgb(dst, &transient_row, &lut_table, image_configuration);
        }
    }

    Ok(())
}

pub fn xyz_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        Some(&empty),
        matrix,
        transfer_function,
    )
}

pub fn xyz_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        Some(&empty),
        matrix,
        transfer_function,
    )
}

pub fn xyz_to_srgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        Some(&empty),
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    )
}

pub fn lab_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        Some(&empty),
        matrix,
        transfer_function,
    )
}

pub fn lab_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        Some(&empty),
        matrix,
        transfer_function,
    )
}

pub fn lab_to_srgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        Some(&empty),
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    )
}

pub fn laba_to_srgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    a_plane: &ImageBuffer<'_, f32>,
) -> Result<(), ColorError> {
    xyz_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        Some(a_plane),
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    )
}

pub fn xyza_to_rgba(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    a_plane: &ImageBuffer<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    xyz_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        Some(a_plane),
        matrix,
        transfer_function,
    )
}

pub fn luv_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        Some(&empty),
        matrix,
        transfer_function,
    )
}

pub fn luv_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        Some(&empty),
        matrix,
        transfer_function,
    )
}

pub fn lch_to_rgb(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        Some(&empty),
        matrix,
        transfer_function,
    )
}

pub fn lch_to_bgr(
    src: &ImageBuffer<'_, f32>,
    dst: &mut ImageBufferMut<'_, u8>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    let empty = ImageBuffer::from_vec(vec![0.; 1], 1, 1, 1, 1)?;
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        Some(&empty),
        matrix,
        transfer_function,
    )
}

#[cfg(test)]
mod tests_xyz_to_channels {
    use super::*;
    use crate::BufferStore;

    fn make_src_f32(
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

    fn make_alpha(data: Vec<f32>, width: u32, height: u32) -> ImageBuffer<'static, f32> {
        ImageBuffer::from_vec(data, width, height, width, 1).unwrap()
    }

    fn assert_approx(a: u8, b: u8, eps: u8, label: &str) {
        assert!(
            (a as i16 - b as i16).abs() <= eps as i16,
            "{label}: got {a}, expected {b}"
        );
    }

    // Convert known linear RGB → colour space for use as round-trip input
    fn linear_rgb_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rgb = Rgb::<f32>::new(r, g, b);
        let xyz = Xyz::from_linear_rgb(rgb, &crate::SRGB_TO_XYZ_D65);
        (xyz.x, xyz.y, xyz.z)
    }

    fn linear_rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rgb = Rgb::<f32>::new(r, g, b);
        let xyz = Xyz::from_linear_rgb(rgb, &crate::SRGB_TO_XYZ_D65);
        let lab = Lab::from_xyz(xyz);
        (lab.l, lab.a, lab.b)
    }

    fn linear_rgb_to_luv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rgb = Rgb::<f32>::new(r, g, b);
        let xyz = Xyz::from_linear_rgb(rgb, &crate::SRGB_TO_XYZ_D65);
        let luv = Luv::from_xyz(xyz);
        (luv.l, luv.u, luv.v)
    }

    fn linear_rgb_to_lch(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rgb = Rgb::<f32>::new(r, g, b);
        let xyz = Xyz::from_linear_rgb(rgb, &crate::SRGB_TO_XYZ_D65);
        let luv = Luv::from_xyz(xyz);
        let lch = LCh::from_luv(luv);
        (lch.l, lch.c, lch.h)
    }

    // ── xyz_to_rgb ────────────────────────────────────────────────────────────

    #[test]
    fn xyz_to_rgb_black_roundtrip() {
        let (x, y, z) = linear_rgb_to_xyz(0.0, 0.0, 0.0);
        let src = make_src_f32(vec![x, y, z], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        xyz_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "R black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "B black");
    }

    #[test]
    fn xyz_to_rgb_white_roundtrip() {
        let (x, y, z) = linear_rgb_to_xyz(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![x, y, z], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        xyz_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "B white");
    }

    #[test]
    fn xyz_to_srgb_matches_xyz_to_rgb() {
        let (x, y, z) = linear_rgb_to_xyz(0.4, 0.6, 0.8);
        let src1 = make_src_f32(vec![x, y, z], 1, 1, 3);
        let src2 = make_src_f32(vec![x, y, z], 1, 1, 3);
        let mut dst1 = make_dst(1, 1, 3);
        let mut dst2 = make_dst(1, 1, 3);
        xyz_to_srgb(&src1, &mut dst1).unwrap();
        xyz_to_rgb(&src2, &mut dst2, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst1.data.borrow(), &*dst2.data.borrow());
    }

    // ── xyz_to_bgr ────────────────────────────────────────────────────────────

    #[test]
    fn xyz_to_bgr_white_roundtrip() {
        let (x, y, z) = linear_rgb_to_xyz(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![x, y, z], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        xyz_to_bgr(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "B white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "R white");
    }

    #[test]
    fn xyz_to_rgb_bgr_agree_on_grey() {
        let (x, y, z) = linear_rgb_to_xyz(0.5, 0.5, 0.5);
        let src_rgb = make_src_f32(vec![x, y, z], 1, 1, 3);
        let src_bgr = make_src_f32(vec![x, y, z], 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        xyz_to_rgb(
            &src_rgb,
            &mut dst_rgb,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        xyz_to_bgr(
            &src_bgr,
            &mut dst_bgr,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert_eq!(
            &*dst_rgb.data.borrow(),
            &*dst_bgr.data.borrow(),
            "grey invariant to channel order"
        );
    }

    #[test]
    fn xyz_to_bgr_channel_order_for_red() {
        let (x, y, z) = linear_rgb_to_xyz(1.0, 0.0, 0.0);
        let src = make_src_f32(vec![x, y, z], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        xyz_to_bgr(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        // BGR: d[0]=B d[1]=G d[2]=R
        assert!(d[0] < 20, "B low for red, got {}", d[0]);
        assert!(d[1] < 20, "G low for red, got {}", d[1]);
        assert!(d[2] > 200, "R dominant for red, got {}", d[2]);
    }

    // ── lab_to_rgb ────────────────────────────────────────────────────────────

    #[test]
    fn lab_to_rgb_black_roundtrip() {
        let (l, a, b) = linear_rgb_to_lab(0.0, 0.0, 0.0);
        let src = make_src_f32(vec![l, a, b], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        lab_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "R black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "B black");
    }

    #[test]
    fn lab_to_rgb_white_roundtrip() {
        let (l, a, b) = linear_rgb_to_lab(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![l, a, b], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        lab_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "B white");
    }

    #[test]
    fn lab_to_srgb_matches_lab_to_rgb() {
        let (l, a, b) = linear_rgb_to_lab(0.3, 0.5, 0.8);
        let src1 = make_src_f32(vec![l, a, b], 1, 1, 3);
        let src2 = make_src_f32(vec![l, a, b], 1, 1, 3);
        let mut dst1 = make_dst(1, 1, 3);
        let mut dst2 = make_dst(1, 1, 3);
        lab_to_srgb(&src1, &mut dst1).unwrap();
        lab_to_rgb(&src2, &mut dst2, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        assert_eq!(&*dst1.data.borrow(), &*dst2.data.borrow());
    }

    #[test]
    fn lab_to_bgr_white_roundtrip() {
        let (l, a, b) = linear_rgb_to_lab(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![l, a, b], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        lab_to_bgr(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "B");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "R");
    }

    #[test]
    fn lab_to_rgb_bgr_agree_on_grey() {
        let (l, a, b) = linear_rgb_to_lab(0.5, 0.5, 0.5);
        let src_rgb = make_src_f32(vec![l, a, b], 1, 1, 3);
        let src_bgr = make_src_f32(vec![l, a, b], 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        lab_to_rgb(
            &src_rgb,
            &mut dst_rgb,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        lab_to_bgr(
            &src_bgr,
            &mut dst_bgr,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert_eq!(&*dst_rgb.data.borrow(), &*dst_bgr.data.borrow());
    }

    // ── laba_to_srgb ──────────────────────────────────────────────────────────

    #[test]
    fn laba_to_srgb_white_opaque() {
        let (l, a, b) = linear_rgb_to_lab(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![l, a, b], 1, 1, 3);
        let alpha = make_alpha(vec![1.0], 1, 1);
        let mut dst = make_dst(1, 1, 4);
        laba_to_srgb(&src, &mut dst, &alpha).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "B");
        assert_approx(d[3], 255, 1, "A");
    }

    #[test]
    fn laba_to_srgb_alpha_denormalized() {
        let (l, a, b) = linear_rgb_to_lab(0.5, 0.5, 0.5);
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src_f32(vec![l, a, b], 1, 1, 3);
            let a_plane = make_alpha(vec![alpha], 1, 1);
            let mut dst = make_dst(1, 1, 4);
            laba_to_srgb(&src, &mut dst, &a_plane).unwrap();
            let d = dst.data.borrow();
            let expected = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected, 1, &format!("alpha={alpha}"));
        }
    }

    #[test]
    fn laba_to_srgb_rgb_encoded_when_alpha_zero() {
        let (l, a, b) = linear_rgb_to_lab(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![l, a, b], 1, 1, 3);
        let a_plane = make_alpha(vec![0.0], 1, 1);
        let mut dst = make_dst(1, 1, 4);
        laba_to_srgb(&src, &mut dst, &a_plane).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R encoded despite alpha=0");
        assert_eq!(d[3], 0, "A must be 0");
    }

    // ── xyza_to_rgba ──────────────────────────────────────────────────────────

    #[test]
    fn xyza_to_rgba_white_opaque() {
        let (x, y, z) = linear_rgb_to_xyz(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![x, y, z], 1, 1, 3);
        let alpha = make_alpha(vec![1.0], 1, 1);
        let mut dst = make_dst(1, 1, 4);
        xyza_to_rgba(
            &src,
            &mut dst,
            &alpha,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R");
        assert_approx(d[1], 255, 2, "G");
        assert_approx(d[2], 255, 2, "B");
        assert_approx(d[3], 255, 1, "A");
    }

    #[test]
    fn xyza_to_rgba_alpha_denormalized() {
        let (x, y, z) = linear_rgb_to_xyz(0.5, 0.5, 0.5);
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let src = make_src_f32(vec![x, y, z], 1, 1, 3);
            let a_plane = make_alpha(vec![alpha], 1, 1);
            let mut dst = make_dst(1, 1, 4);
            xyza_to_rgba(
                &src,
                &mut dst,
                &a_plane,
                &XYZ_TO_SRGB_D65,
                TransferFunction::Srgb,
            )
            .unwrap();
            let d = dst.data.borrow();
            let expected = (alpha * 255.).round() as u8;
            assert_approx(d[3], expected, 1, &format!("alpha={alpha}"));
        }
    }

    // ── luv_to_rgb / luv_to_bgr ───────────────────────────────────────────────

    #[test]
    fn luv_to_rgb_white_roundtrip() {
        let (l, u, v) = linear_rgb_to_luv(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![l, u, v], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        luv_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "B white");
    }

    #[test]
    fn luv_to_rgb_black_roundtrip() {
        let (l, u, v) = linear_rgb_to_luv(0.0, 0.0, 0.0);
        let src = make_src_f32(vec![l, u, v], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        luv_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "R black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "B black");
    }

    #[test]
    fn luv_to_rgb_bgr_agree_on_grey() {
        let (l, u, v) = linear_rgb_to_luv(0.5, 0.5, 0.5);
        let src_rgb = make_src_f32(vec![l, u, v], 1, 1, 3);
        let src_bgr = make_src_f32(vec![l, u, v], 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        luv_to_rgb(
            &src_rgb,
            &mut dst_rgb,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        luv_to_bgr(
            &src_bgr,
            &mut dst_bgr,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert_eq!(&*dst_rgb.data.borrow(), &*dst_bgr.data.borrow());
    }

    // ── lch_to_rgb / lch_to_bgr ───────────────────────────────────────────────

    #[test]
    fn lch_to_rgb_white_roundtrip() {
        let (l, c, h) = linear_rgb_to_lch(1.0, 1.0, 1.0);
        let src = make_src_f32(vec![l, c, h], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        lch_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 255, 2, "R white");
        assert_approx(d[1], 255, 2, "G white");
        assert_approx(d[2], 255, 2, "B white");
    }

    #[test]
    fn lch_to_rgb_black_roundtrip() {
        let (l, c, h) = linear_rgb_to_lch(0.0, 0.0, 0.0);
        let src = make_src_f32(vec![l, c, h], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        lch_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0, 2, "R black");
        assert_approx(d[1], 0, 2, "G black");
        assert_approx(d[2], 0, 2, "B black");
    }

    #[test]
    fn lch_to_rgb_bgr_agree_on_grey() {
        let (l, c, h) = linear_rgb_to_lch(0.5, 0.5, 0.5);
        let src_rgb = make_src_f32(vec![l, c, h], 1, 1, 3);
        let src_bgr = make_src_f32(vec![l, c, h], 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        lch_to_rgb(
            &src_rgb,
            &mut dst_rgb,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        lch_to_bgr(
            &src_bgr,
            &mut dst_bgr,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert_eq!(&*dst_rgb.data.borrow(), &*dst_bgr.data.borrow());
    }

    // ── channel ordering ──────────────────────────────────────────────────────

    #[test]
    fn xyz_to_rgb_channel_order_for_red() {
        let (x, y, z) = linear_rgb_to_xyz(1.0, 0.0, 0.0);
        let src = make_src_f32(vec![x, y, z], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        xyz_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] > 200, "R dominant for red, got {}", d[0]);
        assert!(d[1] < 20, "G low for red, got {}", d[1]);
        assert!(d[2] < 20, "B low for red, got {}", d[2]);
    }

    #[test]
    fn lab_to_rgb_channel_order_for_blue() {
        let (l, a, b) = linear_rgb_to_lab(0.0, 0.0, 1.0);
        let src = make_src_f32(vec![l, a, b], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        lab_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert!(d[0] < 20, "R low for blue, got {}", d[0]);
        assert!(d[1] < 20, "G low for blue, got {}", d[1]);
        assert!(d[2] > 200, "B dominant for blue, got {}", d[2]);
    }

    // ── multi-pixel / multi-row consistency ───────────────────────────────────

    #[test]
    fn xyz_to_rgb_multi_pixel_consistent() {
        let (x, y, z) = linear_rgb_to_xyz(0.4, 0.6, 0.2);
        let src = make_src_f32(vec![x, y, z].repeat(4), 4, 1, 3);
        let mut dst = make_dst(4, 1, 3);
        xyz_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn lab_to_rgb_multi_row_consistent() {
        let (l, a, b) = linear_rgb_to_lab(0.6, 0.3, 0.8);
        let src = make_src_f32(vec![l, a, b].repeat(4), 2, 2, 3);
        let mut dst = make_dst(2, 2, 3);
        lab_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2]];
        for (i, chunk) in d.chunks(3).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    #[test]
    fn laba_to_srgb_multi_pixel_consistent() {
        let (l, a, b) = linear_rgb_to_lab(0.3, 0.5, 0.7);
        let src = make_src_f32(vec![l, a, b].repeat(4), 4, 1, 3);
        let alpha = make_alpha(vec![0.8; 4], 4, 1);
        let mut dst = make_dst(4, 1, 4);
        laba_to_srgb(&src, &mut dst, &alpha).unwrap();
        let d = dst.data.borrow();
        let first = [d[0], d[1], d[2], d[3]];
        for (i, chunk) in d.chunks(4).enumerate() {
            assert_eq!(chunk, first, "px={i}");
        }
    }

    // ── all colour spaces recover same sRGB for known input ───────────────────

    #[test]
    fn all_colour_spaces_agree_on_midgrey() {
        let r = 0.5f32;
        let g = 0.5f32;
        let b = 0.5f32;

        let (x, y, z) = linear_rgb_to_xyz(r, g, b);
        let (l, a, bv) = linear_rgb_to_lab(r, g, b);
        let (lu, u, v) = linear_rgb_to_luv(r, g, b);
        let (lc, c, h) = linear_rgb_to_lch(r, g, b);

        let mut dst_xyz = make_dst(1, 1, 3);
        let mut dst_lab = make_dst(1, 1, 3);
        let mut dst_luv = make_dst(1, 1, 3);
        let mut dst_lch = make_dst(1, 1, 3);

        xyz_to_rgb(
            &make_src_f32(vec![x, y, z], 1, 1, 3),
            &mut dst_xyz,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        lab_to_rgb(
            &make_src_f32(vec![l, a, bv], 1, 1, 3),
            &mut dst_lab,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        luv_to_rgb(
            &make_src_f32(vec![lu, u, v], 1, 1, 3),
            &mut dst_luv,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        lch_to_rgb(
            &make_src_f32(vec![lc, c, h], 1, 1, 3),
            &mut dst_lch,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .unwrap();

        let xyz = dst_xyz.data.borrow();
        let lab = dst_lab.data.borrow();
        let luv = dst_luv.data.borrow();
        let lch = dst_lch.data.borrow();

        for ch in 0..3 {
            assert_approx(lab[ch], xyz[ch], 2, &format!("lab vs xyz ch{ch}"));
            assert_approx(luv[ch], xyz[ch], 2, &format!("luv vs xyz ch{ch}"));
            assert_approx(lch[ch], xyz[ch], 2, &format!("lch vs xyz ch{ch}"));
        }
    }

    // ── error paths ───────────────────────────────────────────────────────────

    #[test]
    fn xyz_to_rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(xyz_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).is_err());
    }

    #[test]
    fn lab_to_rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(lab_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).is_err());
    }

    #[test]
    fn luv_to_rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(luv_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).is_err());
    }

    #[test]
    fn lch_to_rgb_rejects_wrong_channel_count() {
        let src = ImageBuffer::from_vec(vec![0f32; 2], 1, 1, 2, 2).unwrap();
        let mut dst = make_dst(1, 1, 3);
        assert!(lch_to_rgb(&src, &mut dst, &XYZ_TO_SRGB_D65, TransferFunction::Srgb).is_err());
    }
}
