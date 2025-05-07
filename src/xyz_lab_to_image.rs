/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::avx_xyz_to_channels;
use crate::gamma_curves::TransferFunction;
use crate::image::ImageConfiguration;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_xyz_to_channels;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_xyz_to_channels;
use crate::xyz_target::XyzTarget;
use crate::{LCh, Lab, Luv, Xyz, XYZ_TO_SRGB_D65};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::slice;

fn xyz_to_channels<const CHANNELS_CONFIGURATION: u8, const USE_ALPHA: bool, const TARGET: u8>(
    src: &[f32],
    src_stride: u32,
    a_channel: &[f32],
    a_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let source: XyzTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Alpha may be set only on images with alpha");
    }

    let channels = image_configuration.get_channels_count();

    #[allow(clippy::type_complexity)]
    let mut _wide_row_handler: Option<
        unsafe fn(
            usize,
            *const f32,
            usize,
            *const f32,
            usize,
            *mut f32,
            usize,
            u32,
            &[[f32; 3]; 3],
        ) -> usize,
    > = None;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        _wide_row_handler = Some(sse_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        _wide_row_handler = Some(avx_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _wide_row_handler = Some(neon_xyz_to_channels::<CHANNELS_CONFIGURATION, USE_ALPHA, TARGET>);
    }

    let src_slice_safe_align = unsafe {
        slice::from_raw_parts_mut(
            src.as_ptr() as *mut u8,
            src_stride as usize * height as usize,
        )
    };

    let mut lut_table = vec![0u8; 2049];
    for (i, element) in lut_table.iter_mut().enumerate() {
        *element = (transfer_function.gamma(i as f32 * (1. / 2048.0)) * 255.).min(255.) as u8;
    }

    #[cfg(feature = "rayon")]
    {
        if USE_ALPHA {
            let a_slice_safe_align = unsafe {
                slice::from_raw_parts_mut(
                    a_channel.as_ptr() as *mut u8,
                    a_stride as usize * height as usize,
                )
            };

            dst.par_chunks_exact_mut(dst_stride as usize)
                .zip(src_slice_safe_align.par_chunks_exact_mut(src_stride as usize))
                .zip(a_slice_safe_align.par_chunks_exact(a_stride as usize))
                .for_each(|((dst, src), a_channel)| unsafe {
                    let mut _cx = 0usize;

                    let mut transient_row = vec![0f32; width as usize * channels];

                    if let Some(dispatcher) = _wide_row_handler {
                        _cx = dispatcher(
                            _cx,
                            src.as_ptr() as *const f32,
                            0,
                            a_channel.as_ptr() as *const f32,
                            0,
                            transient_row.as_mut_ptr(),
                            0,
                            width,
                            matrix,
                        );
                    }

                    let src_ptr = src.as_ptr() as *mut f32;

                    for x in _cx..width as usize {
                        let src_slice = src_ptr.add(x * 3);
                        let l_x = src_slice.read_unaligned();
                        let l_y = src_slice.add(1).read_unaligned();
                        let l_z = src_slice.add(2).read_unaligned();
                        let rgb = match source {
                            XyzTarget::Lab => {
                                let lab = Lab::new(l_x, l_y, l_z);
                                lab.to_linear_rgb(matrix)
                            }
                            XyzTarget::Xyz => {
                                let xyz = Xyz::new(l_x, l_y, l_z);
                                xyz.to_linear_rgb(matrix)
                            }
                            XyzTarget::Luv => {
                                let luv = Luv::new(l_x, l_y, l_z);
                                luv.to_linear_rgb(matrix)
                            }
                            XyzTarget::Lch => {
                                let lch = LCh::new(l_x, l_y, l_z);
                                lch.to_linear_rgb(matrix)
                            }
                        };

                        let dst = transient_row.get_unchecked_mut((x * channels)..);
                        *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) = rgb.r;
                        *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) = rgb.g;
                        *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) = rgb.b;
                        if image_configuration.has_alpha() {
                            let a_ptr = a_channel.as_ptr() as *const f32;
                            let a_f = a_ptr.add(x).read_unaligned();
                            *dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) =
                                a_f;
                        }
                    }

                    for (dst_chunk, src_chunks) in dst
                        .chunks_exact_mut(channels)
                        .zip(transient_row.chunks_exact(channels))
                    {
                        let r_cast = (src_chunks[image_configuration.get_r_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;
                        let g_cast = (src_chunks[image_configuration.get_g_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;
                        let b_cast = (src_chunks[image_configuration.get_b_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;

                        dst_chunk[image_configuration.get_r_channel_offset()] =
                            *lut_table.get_unchecked(r_cast.min(2048));
                        dst_chunk[image_configuration.get_g_channel_offset()] =
                            *lut_table.get_unchecked(g_cast.min(2048));
                        dst_chunk[image_configuration.get_b_channel_offset()] =
                            *lut_table.get_unchecked(b_cast.min(2048));

                        if image_configuration.has_alpha() {
                            let a_cast = (src_chunks[image_configuration.get_a_channel_offset()]
                                * 255.)
                                .min(255.)
                                .max(0.) as u8;
                            dst_chunk[image_configuration.get_a_channel_offset()] = a_cast;
                        }
                    }
                });
        } else {
            dst.par_chunks_exact_mut(dst_stride as usize)
                .zip(src_slice_safe_align.par_chunks_exact_mut(src_stride as usize))
                .for_each(|(dst, src)| unsafe {
                    let mut _cx = 0usize;

                    let mut transient_row = vec![0f32; width as usize * channels];

                    if let Some(dispatcher) = _wide_row_handler {
                        _cx = dispatcher(
                            _cx,
                            src.as_ptr() as *const f32,
                            0,
                            a_channel.as_ptr(),
                            0,
                            transient_row.as_mut_ptr(),
                            0,
                            width,
                            matrix,
                        );
                    }

                    let src_ptr = src.as_ptr() as *mut f32;

                    for x in _cx..width as usize {
                        let src_slice = src_ptr.add(x * 3);
                        let l_x = src_slice.read_unaligned();
                        let l_y = src_slice.add(1).read_unaligned();
                        let l_z = src_slice.add(2).read_unaligned();
                        let rgb = match source {
                            XyzTarget::Lab => {
                                let lab = Lab::new(l_x, l_y, l_z);
                                lab.to_linear_rgb(matrix)
                            }
                            XyzTarget::Xyz => {
                                let xyz = Xyz::new(l_x, l_y, l_z);
                                xyz.to_linear_rgb(matrix)
                            }
                            XyzTarget::Luv => {
                                let luv = Luv::new(l_x, l_y, l_z);
                                luv.to_linear_rgb(matrix)
                            }
                            XyzTarget::Lch => {
                                let lch = LCh::new(l_x, l_y, l_z);
                                lch.to_linear_rgb(matrix)
                            }
                        };

                        let dst = transient_row.get_unchecked_mut((x * channels)..);
                        *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) = rgb.r;
                        *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) = rgb.g;
                        *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) = rgb.b;
                    }

                    for (dst_chunk, src_chunks) in dst
                        .chunks_exact_mut(channels)
                        .zip(transient_row.chunks_exact(channels))
                    {
                        let r_cast = (src_chunks[image_configuration.get_r_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;
                        let g_cast = (src_chunks[image_configuration.get_g_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;
                        let b_cast = (src_chunks[image_configuration.get_b_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;

                        *dst_chunk.get_unchecked_mut(image_configuration.get_r_channel_offset()) =
                            *lut_table.get_unchecked(r_cast.min(2048));
                        *dst_chunk.get_unchecked_mut(image_configuration.get_g_channel_offset()) =
                            *lut_table.get_unchecked(g_cast.min(2048));
                        *dst_chunk.get_unchecked_mut(image_configuration.get_b_channel_offset()) =
                            *lut_table.get_unchecked(b_cast.min(2048));
                    }
                });
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        if USE_ALPHA {
            let a_slice_safe_align = unsafe {
                slice::from_raw_parts_mut(
                    a_channel.as_ptr() as *mut u8,
                    a_stride as usize * height as usize,
                )
            };

            for ((dst, src), a_channel) in dst
                .chunks_exact_mut(dst_stride as usize)
                .zip(src_slice_safe_align.chunks_exact_mut(src_stride as usize))
                .zip(a_slice_safe_align.chunks_exact(a_stride as usize))
            {
                unsafe {
                    let mut _cx = 0usize;

                    let mut transient_row = vec![0f32; width as usize * channels];

                    if let Some(dispatcher) = _wide_row_handler {
                        _cx = dispatcher(
                            _cx,
                            src.as_ptr() as *const f32,
                            0,
                            a_channel.as_ptr() as *const f32,
                            0,
                            transient_row.as_mut_ptr(),
                            0,
                            width,
                            matrix,
                        );
                    }

                    let src_ptr = src.as_ptr() as *mut f32;

                    for x in _cx..width as usize {
                        let src_slice = src_ptr.add(x * 3);
                        let l_x = src_slice.read_unaligned();
                        let l_y = src_slice.add(1).read_unaligned();
                        let l_z = src_slice.add(2).read_unaligned();
                        let rgb = match source {
                            XyzTarget::Lab => {
                                let lab = Lab::new(l_x, l_y, l_z);
                                lab.to_linear_rgb(matrix)
                            }
                            XyzTarget::Xyz => {
                                let xyz = Xyz::new(l_x, l_y, l_z);
                                xyz.to_linear_rgb(matrix)
                            }
                            XyzTarget::Luv => {
                                let luv = Luv::new(l_x, l_y, l_z);
                                luv.to_linear_rgb(matrix)
                            }
                            XyzTarget::Lch => {
                                let lch = LCh::new(l_x, l_y, l_z);
                                lch.to_linear_rgb(matrix)
                            }
                        };

                        let dst = transient_row.get_unchecked_mut((x * channels)..);
                        *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) = rgb.r;
                        *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) = rgb.g;
                        *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) = rgb.b;
                        if image_configuration.has_alpha() {
                            let a_ptr = a_channel.as_ptr() as *const f32;
                            let a_f = a_ptr.add(x).read_unaligned();
                            *dst.get_unchecked_mut(image_configuration.get_a_channel_offset()) =
                                a_f;
                        }
                    }

                    for (dst_chunk, src_chunks) in dst
                        .chunks_exact_mut(channels)
                        .zip(transient_row.chunks_exact(channels))
                    {
                        let r_cast = (src_chunks[image_configuration.get_r_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;
                        let g_cast = (src_chunks[image_configuration.get_g_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;
                        let b_cast = (src_chunks[image_configuration.get_b_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;

                        dst_chunk[image_configuration.get_r_channel_offset()] =
                            *lut_table.get_unchecked(r_cast.min(2048));
                        dst_chunk[image_configuration.get_g_channel_offset()] =
                            *lut_table.get_unchecked(g_cast.min(2048));
                        dst_chunk[image_configuration.get_b_channel_offset()] =
                            *lut_table.get_unchecked(b_cast.min(2048));

                        if image_configuration.has_alpha() {
                            let a_cast = (src_chunks[image_configuration.get_a_channel_offset()]
                                * 255.)
                                .min(255.)
                                .max(0.) as u8;
                            dst_chunk[image_configuration.get_a_channel_offset()] = a_cast;
                        }
                    }
                }
            }
        } else {
            for (dst, src) in dst
                .chunks_exact_mut(dst_stride as usize)
                .zip(src_slice_safe_align.chunks_exact_mut(src_stride as usize))
            {
                unsafe {
                    let mut _cx = 0usize;

                    let mut transient_row = vec![0f32; width as usize * channels];

                    if let Some(dispatcher) = _wide_row_handler {
                        _cx = dispatcher(
                            _cx,
                            src.as_ptr() as *const f32,
                            0,
                            a_channel.as_ptr(),
                            0,
                            transient_row.as_mut_ptr(),
                            0,
                            width,
                            matrix,
                        );
                    }

                    let src_ptr = src.as_ptr() as *mut f32;

                    for x in _cx..width as usize {
                        let src_slice = src_ptr.add(x * 3);
                        let l_x = src_slice.read_unaligned();
                        let l_y = src_slice.add(1).read_unaligned();
                        let l_z = src_slice.add(2).read_unaligned();
                        let rgb = match source {
                            XyzTarget::Lab => {
                                let lab = Lab::new(l_x, l_y, l_z);
                                lab.to_linear_rgb(matrix)
                            }
                            XyzTarget::Xyz => {
                                let xyz = Xyz::new(l_x, l_y, l_z);
                                xyz.to_linear_rgb(matrix)
                            }
                            XyzTarget::Luv => {
                                let luv = Luv::new(l_x, l_y, l_z);
                                luv.to_linear_rgb(matrix)
                            }
                            XyzTarget::Lch => {
                                let lch = LCh::new(l_x, l_y, l_z);
                                lch.to_linear_rgb(matrix)
                            }
                        };

                        let dst = transient_row.get_unchecked_mut((x * channels)..);
                        *dst.get_unchecked_mut(image_configuration.get_r_channel_offset()) = rgb.r;
                        *dst.get_unchecked_mut(image_configuration.get_g_channel_offset()) = rgb.g;
                        *dst.get_unchecked_mut(image_configuration.get_b_channel_offset()) = rgb.b;
                    }

                    for (dst_chunk, src_chunks) in dst
                        .chunks_exact_mut(channels)
                        .zip(transient_row.chunks_exact(channels))
                    {
                        let r_cast = (src_chunks[image_configuration.get_r_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;
                        let g_cast = (src_chunks[image_configuration.get_g_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;
                        let b_cast = (src_chunks[image_configuration.get_b_channel_offset()]
                            .min(1.)
                            .max(0.)
                            * 2048f32)
                            .round() as usize;

                        dst_chunk[image_configuration.get_r_channel_offset()] =
                            *lut_table.get_unchecked(r_cast.min(2048));
                        dst_chunk[image_configuration.get_g_channel_offset()] =
                            *lut_table.get_unchecked(g_cast.min(2048));
                        dst_chunk[image_configuration.get_b_channel_offset()] =
                            *lut_table.get_unchecked(b_cast.min(2048));
                    }
                }
            }
        }
    }
}

/// This function converts XYZ to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains XYZ data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn xyz_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts XYZ to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains XYZ data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from BGR to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn xyz_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts XYZ to sRGB D65 White point. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains XYZ data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn xyz_to_srgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts LAB to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn lab_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts LAB to BGR. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive BGR data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from BGR to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn lab_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts LAB to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
pub fn lab_to_srgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts LAB with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
pub fn laba_to_srgb(
    src: &[f32],
    src_stride: u32,
    a_plane: &[f32],
    a_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) {
    xyz_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Lab as u8 }>(
        src,
        src_stride,
        a_plane,
        a_stride,
        dst,
        dst_stride,
        width,
        height,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );
}

/// This function converts XYZ with separate alpha channel to RGBA. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `a_plane` - A slice contains Alpha data
/// * `a_stride` - Bytes per row for alpha plane data
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn xyza_to_rgba(
    src: &[f32],
    src_stride: u32,
    a_plane: &[f32],
    a_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    xyz_to_channels::<{ ImageConfiguration::Rgba as u8 }, true, { XyzTarget::Xyz as u8 }>(
        src,
        src_stride,
        a_plane,
        a_stride,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts LUV to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn luv_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts LUV to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn luv_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Luv as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts LCH to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn lch_to_rgb(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Rgb as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}

/// This function converts LCH to RGB. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains LAB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive RGB data
/// * `dst_stride` - Bytes per row for dst data
/// * `matrix` - Transformation matrix from RGB to XYZ. If you don't have specific just pick `XYZ_TO_SRGB_D65`
/// * `transfer_function` - Transfer function. If you don't have specific pick `Srgb`
pub fn lch_to_bgr(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    matrix: &[[f32; 3]; 3],
    transfer_function: TransferFunction,
) {
    let empty_vec = vec![];
    xyz_to_channels::<{ ImageConfiguration::Bgr as u8 }, false, { XyzTarget::Lch as u8 }>(
        src,
        src_stride,
        &empty_vec,
        0,
        dst,
        dst_stride,
        width,
        height,
        matrix,
        transfer_function,
    );
}
