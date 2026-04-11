/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
use crate::xyz_target::XyzTarget;
use crate::{
    BufferStore, ColorError, ImageBuffer, ImageBufferMut, LCh, Lab, Luv, Rgb, SRGB_TO_XYZ_D65, Xyz,
};

#[allow(clippy::type_complexity)]
fn channels_to_xyz<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool, const TARGET: u8>(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    a_plane: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    src.validate()?;
    dst.validate()?;
    dst.try_match_immutable(src)?;
    if USE_ALPHA {
        dst.try_match(a_plane)?;
    }
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    if dst.channels != 3 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    if USE_ALPHA && a_plane.channels != 1 {
        return Err(ColorError::UnsupportedChannelsCount(a_plane.channels));
    }
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.channel_count();

    let mut _wide_row_handler: Option<
        unsafe fn(usize, &[f32], &mut [f32], &mut [f32], u32, &[[f32; 3]; 3]) -> usize,
    > = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::neon::neon_channels_to_xyz_or_lab;
        _wide_row_handler =
            Some(neon_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        use crate::sse::sse_channels_to_xyz_or_lab;
        _wide_row_handler =
            Some(sse_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        use crate::avx::avx2_image_to_xyz_lab;
        _wide_row_handler =
            Some(avx2_image_to_xyz_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

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
    let width = src.width;

    for (dst, src) in dst
        .data
        .borrow_mut()
        .chunks_mut(dst_stride)
        .zip(src.data.chunks(src.stride()))
    {
        let src = &src[..src_r_width as usize];
        let dst = &mut dst[..dst_r_width as usize];

        let mut _cx = 0usize;

        let mut transient_row = vec![0f32; width as usize * channels];

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

        if let Some(dispatcher) = _wide_row_handler {
            _cx = unsafe {
                dispatcher(
                    _cx,
                    &transient_row,
                    dst,
                    a_plane.data.borrow_mut(),
                    width,
                    matrix,
                )
            };
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
            for ((dst, src), a_dst) in dst
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(transient_row.as_chunks::<4>().0.iter())
                .zip(a_plane.data.borrow_mut())
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
                *a_dst = src[3];
            }
        }
    }
    Ok(())
}

/// This function converts RGB to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_xyz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

/// This function converts BGR to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from BGR to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgr_to_xyz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

/// This function converts sRGB D65 to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn srgb_to_xyz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    )
}

/// This function converts RGB to CIE L*ab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_lab(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

/// This function converts RGBA to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgba_to_xyz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

/// This function converts sRGB RGBA D65 to XYZ. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGBA to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn srgba_to_xyz(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    )
}

/// This function converts RGBA to XYZ with preserving and linearizing alpha channels. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgba_to_xyza(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    a_plane: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        a_plane,
        matrix,
        transfer_function,
    )
}

/// This function converts RGBA to XYZ with preserving and linearizing alpha channels. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive XYZ data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
pub fn srgba_to_xyza(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    a_plane: &mut ImageBufferMut<'_, f32>,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        a_plane,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    )
}

/// This function converts RGBA to CIE L*ab against D65 white point without alpha. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgba_to_lab(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    a_plane: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        a_plane,
        matrix,
        transfer_function,
    )
}

/// This function converts RGBA to CIE L*ab against D65 white point and preserving and normalizing alpha channels. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGBA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `a_plane` - A mutable slice to receive XYZ data
/// * `a_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgba_to_laba(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    a_plane: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        a_plane,
        matrix,
        transfer_function,
    )
}

/// This function converts BGRA to CIE L*ab against D65 white point and preserving and linearizing alpha channels. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGRA data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgra_to_laba(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    a_plane: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Bgra as u8 }, true, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        a_plane,
        matrix,
        transfer_function,
    )
}

/// This function converts BGR to CIE L*ab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgr_to_lab(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

/// This function converts RGB to CIE L*uv against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_luv(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

/// This function converts BGR to CIE L*ab against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains BGR data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgr_to_luv(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

/// This function converts RGB to CIE L\*C\*h against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn rgb_to_lch(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

/// This function converts BGR to CIE L\*C\*h against D65 white point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive LAB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `SRGB_TO_XYZ_D65`
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn bgr_to_lch(
    src: &ImageBuffer<'_, u8>,
    dst: &mut ImageBufferMut<'_, f32>,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) -> Result<(), ColorError> {
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(vec![0.; 1]), 1, 1, 1)?,
        matrix,
        transfer_function,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_src(data: Vec<u8>, width: u32, height: u32, channels: u32) -> ImageBuffer<'static, u8> {
        ImageBuffer::from_vec(data, width, height, channels).unwrap()
    }

    fn make_dst(width: u32, height: u32, channels: u32) -> ImageBufferMut<'static, f32> {
        let len = (width * height * channels) as usize;
        ImageBufferMut::new(BufferStore::Owned(vec![0f32; len]), width, height, channels).unwrap()
    }

    fn make_alpha(width: u32, height: u32) -> ImageBufferMut<'static, f32> {
        let len = (width * height) as usize;
        ImageBufferMut::new(BufferStore::Owned(vec![0f32; len]), width, height, 1).unwrap()
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn assert_approx(a: f32, b: f32, eps: f32, label: &str) {
        assert!((a - b).abs() < eps, "{label}: expected {b}, got {a}");
    }

    // ── black / white sanity ──────────────────────────────────────────────────

    #[test]
    fn rgb_to_xyz_black() {
        let src = make_src(vec![0u8; 3], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_xyz(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0.0, 1e-5, "X");
        assert_approx(d[1], 0.0, 1e-5, "Y");
        assert_approx(d[2], 0.0, 1e-5, "Z");
    }

    #[test]
    fn rgb_to_xyz_white() {
        let src = make_src(vec![255u8; 3], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_xyz(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        // D65 white point
        assert_approx(d[0], 0.9505, 1e-3, "X");
        assert_approx(d[1], 1.0000, 1e-3, "Y");
        assert_approx(d[2], 1.0890, 1e-3, "Z");
    }

    #[test]
    fn bgr_to_xyz_white() {
        // BGR white is identical to RGB white
        let src = make_src(vec![255u8; 3], 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        rgb_to_xyz(&src, &mut dst_rgb, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        bgr_to_xyz(&src, &mut dst_bgr, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let r = dst_rgb.data.borrow();
        let b = dst_bgr.data.borrow();
        for i in 0..3 {
            assert_approx(b[i], r[i], 1e-5, &format!("channel {i}"));
        }
    }

    // ── sRGB convenience wrappers ─────────────────────────────────────────────

    #[test]
    fn srgb_to_xyz_matches_rgb_to_xyz() {
        let data = vec![100u8, 150u8, 200u8];
        let src = make_src(data.clone(), 1, 1, 3);
        let src2 = make_src(data, 1, 1, 3);
        let mut dst1 = make_dst(1, 1, 3);
        let mut dst2 = make_dst(1, 1, 3);
        srgb_to_xyz(&src, &mut dst1).unwrap();
        rgb_to_xyz(&src2, &mut dst2, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let d1 = dst1.data.borrow();
        let d2 = dst2.data.borrow();
        for i in 0..3 {
            assert_approx(d1[i], d2[i], 1e-6, &format!("channel {i}"));
        }
    }

    #[test]
    fn srgba_to_xyz_matches_rgba_to_xyz() {
        let data = vec![100u8, 150u8, 200u8, 255u8];
        let src = make_src(data.clone(), 1, 1, 4);
        let src2 = make_src(data, 1, 1, 4);
        let mut dst1 = make_dst(1, 1, 3);
        let mut dst2 = make_dst(1, 1, 3);
        srgba_to_xyz(&src, &mut dst1).unwrap();
        rgba_to_xyz(&src2, &mut dst2, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let d1 = dst1.data.borrow();
        let d2 = dst2.data.borrow();
        for i in 0..3 {
            assert_approx(d1[i], d2[i], 1e-6, &format!("channel {i}"));
        }
    }

    // ── Lab ───────────────────────────────────────────────────────────────────

    #[test]
    fn rgb_to_lab_white() {
        let src = make_src(vec![255u8; 3], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_lab(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 100.0, 0.5, "L*");
        assert_approx(d[1], 0.0, 0.5, "a*");
        assert_approx(d[2], 0.0, 0.5, "b*");
    }

    #[test]
    fn rgb_to_lab_black() {
        let src = make_src(vec![0u8; 3], 1, 1, 3);
        let mut dst = make_dst(1, 1, 3);
        rgb_to_lab(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        assert_approx(d[0], 0.0, 0.1, "L*");
    }

    #[test]
    fn bgr_to_lab_matches_rgb_to_lab_for_grey() {
        // Grey pixels are BGR-invariant
        let data = vec![128u8, 128u8, 128u8];
        let src_rgb = make_src(data.clone(), 1, 1, 3);
        let src_bgr = make_src(data, 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        rgb_to_lab(
            &src_rgb,
            &mut dst_rgb,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgr_to_lab(
            &src_bgr,
            &mut dst_bgr,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        let r = dst_rgb.data.borrow();
        let b = dst_bgr.data.borrow();
        for i in 0..3 {
            assert_approx(b[i], r[i], 1e-4, &format!("channel {i}"));
        }
    }

    // ── RGBA → XYZA (alpha preservation) ─────────────────────────────────────

    #[test]
    fn rgba_to_xyza_alpha_passthrough() {
        // Alpha = 128 → normalised to 128/255 ≈ 0.502
        let src = make_src(vec![255, 255, 255, 128], 1, 1, 4);
        let mut dst = make_dst(1, 1, 3);
        let mut alpha = make_alpha(1, 1);
        rgba_to_xyza(
            &src,
            &mut dst,
            &mut alpha,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        let a = alpha.data.borrow();
        assert_approx(a[0], 128.0 / 255.0, 1e-4, "alpha");
    }

    #[test]
    fn srgba_to_xyza_alpha_passthrough() {
        let src = make_src(vec![0, 0, 0, 64], 1, 1, 4);
        let mut dst = make_dst(1, 1, 3);
        let mut alpha = make_alpha(1, 1);
        srgba_to_xyza(&src, &mut dst, &mut alpha).unwrap();
        let a = alpha.data.borrow();
        assert_approx(a[0], 64.0 / 255.0, 1e-4, "alpha");
    }

    #[test]
    fn bgr_to_luv_matches_rgb_for_grey() {
        let data = vec![200u8, 200u8, 200u8];
        let src_rgb = make_src(data.clone(), 1, 1, 3);
        let src_bgr = make_src(data, 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        rgb_to_luv(
            &src_rgb,
            &mut dst_rgb,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgr_to_luv(
            &src_bgr,
            &mut dst_bgr,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        let r = dst_rgb.data.borrow();
        let b = dst_bgr.data.borrow();
        for i in 0..3 {
            assert_approx(b[i], r[i], 1e-4, &format!("channel {i}"));
        }
    }

    #[test]
    fn bgr_to_lch_matches_rgb_for_grey() {
        let data = vec![100u8, 100u8, 100u8];
        let src_rgb = make_src(data.clone(), 1, 1, 3);
        let src_bgr = make_src(data, 1, 1, 3);
        let mut dst_rgb = make_dst(1, 1, 3);
        let mut dst_bgr = make_dst(1, 1, 3);
        rgb_to_lch(
            &src_rgb,
            &mut dst_rgb,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        bgr_to_lch(
            &src_bgr,
            &mut dst_bgr,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .unwrap();
        let r = dst_rgb.data.borrow();
        let b = dst_bgr.data.borrow();
        for i in 0..3 {
            assert_approx(b[i], r[i], 1e-4, &format!("channel {i}"));
        }
    }

    // ── multi-pixel / stride ──────────────────────────────────────────────────

    #[test]
    fn rgb_to_xyz_multi_pixel_consistent() {
        // 3 identical pixels — all outputs must be identical
        let pixel = [100u8, 149u8, 237u8]; // cornflower blue-ish
        let src = make_src(pixel.repeat(3), 3, 1, 3);
        let mut dst = make_dst(3, 1, 3);
        rgb_to_xyz(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        for px in 1..3 {
            for ch in 0..3 {
                assert_approx(d[px * 3 + ch], d[ch], 1e-5, &format!("px{px} ch{ch}"));
            }
        }
    }

    #[test]
    fn rgb_to_lab_multi_row() {
        // 1x2 image, both rows identical white → both Lab rows must match
        let src = make_src(vec![255u8; 6], 1, 2, 3);
        let mut dst = make_dst(1, 2, 3);
        rgb_to_lab(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb).unwrap();
        let d = dst.data.borrow();
        for ch in 0..3 {
            assert_approx(d[ch], d[3 + ch], 1e-5, &format!("row mismatch ch{ch}"));
        }
    }

    // ── error paths ───────────────────────────────────────────────────────────

    #[test]
    fn wrong_channel_count_returns_error() {
        // 2-channel source is unsupported
        let src = make_src(vec![0u8; 2], 1, 1, 2);
        let mut dst = make_dst(1, 1, 3);
        let result = rgb_to_xyz(&src, &mut dst, &SRGB_TO_XYZ_D65, TransferFunction::Srgb);
        assert!(result.is_err());
    }
}
