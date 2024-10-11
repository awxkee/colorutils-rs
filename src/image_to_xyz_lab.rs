/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx2_image_to_xyz_lab;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_channels_to_xyz_or_lab;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_channels_to_xyz_or_lab;
use crate::xyz_target::XyzTarget;
use crate::{LCh, Lab, Luv, Rgb, Xyz, SRGB_TO_XYZ_D65};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

#[allow(clippy::type_complexity)]
fn channels_to_xyz<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool, const TARGET: u8>(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_channel: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let target: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    let mut _wide_row_handler: Option<
        unsafe fn(
            usize,
            *const f32,
            usize,
            u32,
            *mut f32,
            usize,
            *mut f32,
            usize,
            &[[f32; 3]; 3],
        ) -> usize,
    > = None;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler =
            Some(neon_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler =
            Some(sse_channels_to_xyz_or_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handler =
            Some(avx2_image_to_xyz_lab::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    let mut lut_table = vec![0f32; 256];
    for i in 0..256 {
        lut_table[i] = transfer_function.linearize(i as f32 * (1. / 255.0));
    }

    let dst_slice_safe_align = unsafe {
        slice::from_raw_parts_mut(
            dst.as_mut_ptr() as *mut u8,
            dst_stride as usize * height as usize,
        )
    };

    #[cfg(feature = "rayon")]
    {
        if USE_ALPHA {
            let a_slice_safe_align = unsafe {
                slice::from_raw_parts_mut(
                    a_channel.as_mut_ptr() as *mut u8,
                    a_stride as usize * height as usize,
                )
            };

            dst_slice_safe_align
                .par_chunks_exact_mut(dst_stride as usize)
                .zip(src.par_chunks_exact(src_stride as usize))
                .zip(a_slice_safe_align.par_chunks_exact_mut(a_stride as usize))
                .for_each(|((dst, src), a_channel)| unsafe {
                    let mut _cx = 0usize;

                    let mut transient_row = vec![0f32; width as usize * channels];

                    for (dst_chunk, src_chunks) in transient_row
                        .chunks_exact_mut(channels)
                        .zip(src.chunks_exact(channels))
                    {
                        dst_chunk[image_configuration.get_r_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_r_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_g_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_g_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_b_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_b_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_a_channel_offset()] =
                            src_chunks[image_configuration.get_a_channel_offset()] as f32
                                * (1. / 255.0);
                    }

                    if let Some(dispatcher) = _wide_row_handler {
                        _cx = dispatcher(
                            _cx,
                            transient_row.as_ptr(),
                            0,
                            width,
                            dst.as_mut_ptr() as *mut f32,
                            0,
                            a_channel.as_mut_ptr() as *mut f32,
                            0,
                            matrix,
                        );
                    }

                    let dst_ptr = dst.as_mut_ptr().add(0) as *mut f32;

                    for x in _cx..width as usize {
                        let px = x * channels;
                        let src = transient_row.get_unchecked(px..);
                        let r = *src.get_unchecked(image_configuration.get_r_channel_offset());
                        let g = *src.get_unchecked(image_configuration.get_g_channel_offset());
                        let b = *src.get_unchecked(image_configuration.get_b_channel_offset());

                        let rgb = Rgb::<f32>::new(r, g, b);
                        let ptr = dst_ptr.add(x * 3);

                        let xyz = Xyz::from_linear_rgb(rgb, matrix);

                        match target {
                            XyzTarget::Lab => {
                                let lab = Lab::from_xyz(xyz);
                                ptr.write_unaligned(lab.l);
                                ptr.add(1).write_unaligned(lab.a);
                                ptr.add(2).write_unaligned(lab.b);
                            }
                            XyzTarget::Xyz => {
                                ptr.write_unaligned(xyz.x);
                                ptr.add(1).write_unaligned(xyz.y);
                                ptr.add(2).write_unaligned(xyz.z);
                            }
                            XyzTarget::Luv => {
                                let luv = Luv::from_xyz(xyz);
                                ptr.write_unaligned(luv.l);
                                ptr.add(1).write_unaligned(luv.u);
                                ptr.add(2).write_unaligned(luv.v);
                            }
                            XyzTarget::Lch => {
                                let luv = Luv::from_xyz(xyz);
                                let lch = LCh::from_luv(luv);
                                ptr.write_unaligned(lch.l);
                                ptr.add(1).write_unaligned(lch.c);
                                ptr.add(2).write_unaligned(lch.h);
                            }
                        }

                        if USE_ALPHA && image_configuration.has_alpha() {
                            let a = *src.get_unchecked(image_configuration.get_a_channel_offset());
                            let a_ptr = a_channel.as_mut_ptr() as *mut f32;
                            a_ptr.add(x).write_unaligned(a);
                        }
                    }
                });
        } else {
            dst_slice_safe_align
                .par_chunks_exact_mut(dst_stride as usize)
                .zip(src.par_chunks_exact(src_stride as usize))
                .for_each(|(dst, src)| unsafe {
                    let mut _cx = 0usize;

                    let mut transient_row = vec![0f32; width as usize * channels];

                    for (dst_chunk, src_chunks) in transient_row
                        .chunks_exact_mut(channels)
                        .zip(src.chunks_exact(channels))
                    {
                        dst_chunk[image_configuration.get_r_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_r_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_g_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_g_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_b_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_b_channel_offset()] as usize,
                            );
                    }

                    if let Some(dispatcher) = _wide_row_handler {
                        _cx = dispatcher(
                            _cx,
                            transient_row.as_ptr(),
                            0,
                            width,
                            dst.as_mut_ptr() as *mut f32,
                            0,
                            std::ptr::null_mut(),
                            0,
                            matrix,
                        );
                    }

                    let dst_ptr = dst.as_mut_ptr().add(0) as *mut f32;

                    for x in _cx..width as usize {
                        let px = x * channels;
                        let src = transient_row.get_unchecked(px..);
                        let r = *src.get_unchecked(image_configuration.get_r_channel_offset());
                        let g = *src.get_unchecked(image_configuration.get_g_channel_offset());
                        let b = *src.get_unchecked(image_configuration.get_b_channel_offset());

                        let rgb = Rgb::<f32>::new(r, g, b);
                        let ptr = dst_ptr.add(x * 3);

                        let xyz = Xyz::from_linear_rgb(rgb, matrix);

                        match target {
                            XyzTarget::Lab => {
                                let lab = Lab::from_xyz(xyz);
                                ptr.write_unaligned(lab.l);
                                ptr.add(1).write_unaligned(lab.a);
                                ptr.add(2).write_unaligned(lab.b);
                            }
                            XyzTarget::Xyz => {
                                ptr.write_unaligned(xyz.x);
                                ptr.add(1).write_unaligned(xyz.y);
                                ptr.add(2).write_unaligned(xyz.z);
                            }
                            XyzTarget::Luv => {
                                let luv = Luv::from_xyz(xyz);
                                ptr.write_unaligned(luv.l);
                                ptr.add(1).write_unaligned(luv.u);
                                ptr.add(2).write_unaligned(luv.v);
                            }
                            XyzTarget::Lch => {
                                let luv = Luv::from_xyz(xyz);
                                let lch = LCh::from_luv(luv);
                                ptr.write_unaligned(lch.l);
                                ptr.add(1).write_unaligned(lch.c);
                                ptr.add(2).write_unaligned(lch.h);
                            }
                        }
                    }
                });
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        if USE_ALPHA {
            let a_slice_safe_align = unsafe {
                slice::from_raw_parts_mut(
                    a_channel.as_mut_ptr() as *mut u8,
                    a_stride as usize * height as usize,
                )
            };

            for ((dst, src), a_channel) in dst_slice_safe_align
                .chunks_exact_mut(dst_stride as usize)
                .zip(src.chunks_exact(src_stride as usize))
                .zip(a_slice_safe_align.chunks_exact_mut(a_stride as usize))
            {
                unsafe {
                    let mut _cx = 0usize;

                    let mut transient_row = vec![0f32; width as usize * channels];

                    for (dst_chunk, src_chunks) in transient_row
                        .chunks_exact_mut(channels)
                        .zip(src.chunks_exact(channels))
                    {
                        dst_chunk[image_configuration.get_r_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_r_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_g_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_g_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_b_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_b_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_a_channel_offset()] =
                            src_chunks[image_configuration.get_a_channel_offset()] as f32
                                * (1. / 255.0);
                    }

                    if let Some(dispatcher) = _wide_row_handler {
                        _cx = dispatcher(
                            _cx,
                            transient_row.as_ptr(),
                            0,
                            width,
                            dst.as_mut_ptr() as *mut f32,
                            0,
                            a_channel.as_mut_ptr() as *mut f32,
                            0,
                            matrix,
                        );
                    }

                    let dst_ptr = dst.as_mut_ptr().add(0) as *mut f32;

                    for x in _cx..width as usize {
                        let px = x * channels;
                        let src = transient_row.get_unchecked(px..);
                        let r = *src.get_unchecked(image_configuration.get_r_channel_offset());
                        let g = *src.get_unchecked(image_configuration.get_g_channel_offset());
                        let b = *src.get_unchecked(image_configuration.get_b_channel_offset());

                        let rgb = Rgb::<f32>::new(r, g, b);
                        let ptr = dst_ptr.add(x * 3);

                        let xyz = Xyz::from_linear_rgb(rgb, matrix);

                        match target {
                            XyzTarget::Lab => {
                                let lab = Lab::from_xyz(xyz);
                                ptr.write_unaligned(lab.l);
                                ptr.add(1).write_unaligned(lab.a);
                                ptr.add(2).write_unaligned(lab.b);
                            }
                            XyzTarget::Xyz => {
                                ptr.write_unaligned(xyz.x);
                                ptr.add(1).write_unaligned(xyz.y);
                                ptr.add(2).write_unaligned(xyz.z);
                            }
                            XyzTarget::Luv => {
                                let luv = Luv::from_xyz(xyz);
                                ptr.write_unaligned(luv.l);
                                ptr.add(1).write_unaligned(luv.u);
                                ptr.add(2).write_unaligned(luv.v);
                            }
                            XyzTarget::Lch => {
                                let luv = Luv::from_xyz(xyz);
                                let lch = LCh::from_luv(luv);
                                ptr.write_unaligned(lch.l);
                                ptr.add(1).write_unaligned(lch.c);
                                ptr.add(2).write_unaligned(lch.h);
                            }
                        }

                        if USE_ALPHA && image_configuration.has_alpha() {
                            let a = *src.get_unchecked(image_configuration.get_a_channel_offset());
                            let a_ptr = a_channel.as_mut_ptr() as *mut f32;
                            a_ptr.add(x).write_unaligned(a);
                        }
                    }
                }
            }
        } else {
            for (dst, src) in dst_slice_safe_align
                .chunks_exact_mut(dst_stride as usize)
                .zip(src.chunks_exact(src_stride as usize))
            {
                unsafe {
                    let mut _cx = 0usize;

                    let mut transient_row = vec![0f32; width as usize * channels];

                    for (dst_chunk, src_chunks) in transient_row
                        .chunks_exact_mut(channels)
                        .zip(src.chunks_exact(channels))
                    {
                        dst_chunk[image_configuration.get_r_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_r_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_g_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_g_channel_offset()] as usize,
                            );
                        dst_chunk[image_configuration.get_b_channel_offset()] = *lut_table
                            .get_unchecked(
                                src_chunks[image_configuration.get_b_channel_offset()] as usize,
                            );
                    }

                    if let Some(dispatcher) = _wide_row_handler {
                        _cx = dispatcher(
                            _cx,
                            transient_row.as_ptr(),
                            0,
                            width,
                            dst.as_mut_ptr() as *mut f32,
                            0,
                            std::ptr::null_mut(),
                            0,
                            matrix,
                        );
                    }

                    let dst_ptr = dst.as_mut_ptr().add(0) as *mut f32;

                    for x in _cx..width as usize {
                        let px = x * channels;
                        let src = transient_row.get_unchecked(px..);
                        let r = *src.get_unchecked(image_configuration.get_r_channel_offset());
                        let g = *src.get_unchecked(image_configuration.get_g_channel_offset());
                        let b = *src.get_unchecked(image_configuration.get_b_channel_offset());

                        let rgb = Rgb::<f32>::new(r, g, b);
                        let ptr = dst_ptr.add(x * 3);

                        let xyz = Xyz::from_linear_rgb(rgb, matrix);

                        match target {
                            XyzTarget::Lab => {
                                let lab = Lab::from_xyz(xyz);
                                ptr.write_unaligned(lab.l);
                                ptr.add(1).write_unaligned(lab.a);
                                ptr.add(2).write_unaligned(lab.b);
                            }
                            XyzTarget::Xyz => {
                                ptr.write_unaligned(xyz.x);
                                ptr.add(1).write_unaligned(xyz.y);
                                ptr.add(2).write_unaligned(xyz.z);
                            }
                            XyzTarget::Luv => {
                                let luv = Luv::from_xyz(xyz);
                                ptr.write_unaligned(luv.l);
                                ptr.add(1).write_unaligned(luv.u);
                                ptr.add(2).write_unaligned(luv.v);
                            }
                            XyzTarget::Lch => {
                                let luv = Luv::from_xyz(xyz);
                                let lch = LCh::from_luv(luv);
                                ptr.write_unaligned(lch.l);
                                ptr.add(1).write_unaligned(lch.c);
                                ptr.add(2).write_unaligned(lch.h);
                            }
                        }
                    }
                }
            }
        }
    }
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
) {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    a_plane: &mut [f32],
    a_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    channels_to_xyz::<{ ImageConfiguration::Bgra as u8 }, true, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        a_plane,
        a_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
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
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let mut empty_vec = vec![];
    channels_to_xyz::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        dst,
        dst_stride,
        &mut empty_vec,
        0,
        width,
        height,
        matrix,
        transfer_function,
    );
}
