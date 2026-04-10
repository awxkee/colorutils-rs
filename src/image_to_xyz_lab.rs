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
    BufferStore, ColorError, ImageBuffer, ImageBufferMut, LCh, Lab, Luv, Rgb, Xyz, SRGB_TO_XYZ_D65,
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
    dst.try_match_immutable_with_channels(src)?;
    if USE_ALPHA {
        dst.try_match(a_plane)?;
    }
    if src.channels != 3 && src.channels != 4 {
        return Err(ColorError::UnsupportedChannelsCount(src.channels));
    }
    if USE_ALPHA {
        if a_plane.channels != 1 {
            return Err(ColorError::UnsupportedChannelsCount(a_plane.channels));
        }
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
                dst[0] = lut_table[src[0] as usize];
                dst[1] = lut_table[src[1] as usize];
                dst[2] = lut_table[src[2] as usize];
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
                dst[0] = lut_table[src[0] as usize];
                dst[1] = lut_table[src[1] as usize];
                dst[2] = lut_table[src[2] as usize];
                dst[3] = a_lut_table[src[3] as usize];
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
    let empty_vec = vec![0.; dst.width as usize * dst.height as usize];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(
            BufferStore::Owned(empty_vec),
            dst.width,
            dst.height,
            dst.width,
            1,
        )?,
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
    let empty_vec = vec![0.; dst.width as usize * dst.height as usize];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(
            BufferStore::Owned(empty_vec),
            dst.width,
            dst.height,
            dst.width,
            1,
        )?,
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
    let empty_vec = vec![0.; dst.width as usize * dst.height as usize];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(
            BufferStore::Owned(empty_vec),
            dst.width,
            dst.height,
            dst.width,
            1,
        )?,
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
    let empty_vec = vec![0.; 1];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(empty_vec), 1, 1, 1, 1)?,
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
    let empty_vec = vec![0.; 1];
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(empty_vec), 1, 1, 1, 1)?,
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
    let empty_vec = vec![0.; 1];
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(BufferStore::Owned(empty_vec), 1, 1, 1, 1)?,
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
    let empty_vec = vec![0.; dst.width as usize * dst.height as usize];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(
            BufferStore::Owned(empty_vec),
            dst.width,
            dst.height,
            dst.width,
            1,
        )?,
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
    let empty_vec = vec![0.; dst.width as usize * dst.height as usize];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(
            BufferStore::Owned(empty_vec),
            dst.width,
            dst.height,
            dst.width,
            1,
        )?,
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
    let empty_vec = vec![0.; dst.width as usize * dst.height as usize];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(
            BufferStore::Owned(empty_vec),
            dst.width,
            dst.height,
            dst.width,
            1,
        )?,
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
    let empty_vec = vec![0.; dst.width as usize * dst.height as usize];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(
            BufferStore::Owned(empty_vec),
            dst.width,
            dst.height,
            dst.width,
            1,
        )?,
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
    let empty_vec = vec![0.; dst.width as usize * dst.height as usize];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        dst,
        &mut ImageBufferMut::new(
            BufferStore::Owned(empty_vec),
            dst.width,
            dst.height,
            dst.width,
            1,
        )?,
        matrix,
        transfer_function,
    )
}
