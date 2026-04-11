/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::{ColorError, ImageBuffer, ImageBufferMut};

/// Expands RGB to RGBA.
pub fn rgb_to_rgba(
    src: &ImageBuffer<u8>,
    dst: &mut ImageBufferMut<u8>,
    default_alpha: u8,
) -> Result<(), ColorError> {
    src.validate()?;
    dst.validate()?;
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

    dst.try_match_immutable(src)?;

    let src_working_width = src.width * src.channels;
    let dst_working_width = dst.width * dst.channels;
    let dst_stride = dst.stride();
    for (src, dst) in src
        .data
        .chunks(src.stride())
        .zip(dst.data.borrow_mut().chunks_mut(dst_stride))
    {
        let src = &src[..src_working_width as usize];
        let dst = &mut dst[..dst_working_width as usize];
        for (src, dst) in src
            .as_chunks::<3>()
            .0
            .iter()
            .zip(dst.as_chunks_mut::<4>().0.iter_mut())
        {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = default_alpha;
        }
    }

    Ok(())
}

/// Expands BGR to BGRA.
pub fn bgr_to_bgra(
    src: &ImageBuffer<u8>,
    dst: &mut ImageBufferMut<u8>,
    default_alpha: u8,
) -> Result<(), ColorError> {
    rgb_to_rgba(src, dst, default_alpha)
}
