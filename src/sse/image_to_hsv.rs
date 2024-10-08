/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::image::ImageConfiguration;
use crate::image_to_hsv_support::HsvTarget;
use crate::sse::color::{sse_rgb_to_hsl, sse_rgb_to_hsv};
use crate::sse::{
    sse_deinterleave_rgb, sse_deinterleave_rgba, sse_interleave_rgb_epi16,
    sse_interleave_rgba_epi16,
};
use crate::{load_u8_and_deinterleave, store_and_interleave_v3_u16, store_and_interleave_v4_u16};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_channels_to_hsv_u16<
    const CHANNELS_CONFIGURATION: u8,
    const USE_ALPHA: bool,
    const TARGET: u8,
>(
    start_cx: usize,
    src: *const u8,
    src_offset: usize,
    width: u32,
    dst: *mut u16,
    dst_offset: usize,
    scale: f32,
) -> usize {
    let target: HsvTarget = TARGET.into();
    let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
    let mut cx = start_cx;
    if USE_ALPHA && !image_configuration.has_alpha() {
        panic!("Use alpha flag used on image without alpha");
    }

    let channels = image_configuration.get_channels_count();

    let v_scale = _mm_set1_ps(scale);

    let dst_ptr = (dst as *mut u8).add(dst_offset) as *mut u16;

    while cx + 16 < width as usize {
        let src_ptr = src.add(src_offset + cx * channels);
        let (r_chan, g_chan, b_chan, a_chan) =
            load_u8_and_deinterleave!(src_ptr, image_configuration);

        let zeros = _mm_setzero_si128();

        let r_low = _mm_unpacklo_epi8(r_chan, zeros);
        let g_low = _mm_unpacklo_epi8(g_chan, zeros);
        let b_low = _mm_unpacklo_epi8(b_chan, zeros);

        let r_low_low = _mm_unpacklo_epi16(r_low, zeros);
        let g_low_low = _mm_unpacklo_epi16(g_low, zeros);
        let b_low_low = _mm_unpacklo_epi16(b_low, zeros);

        let (x_low_low, y_low_low, z_low_low) = match target {
            HsvTarget::Hsv => sse_rgb_to_hsv(r_low_low, g_low_low, b_low_low, v_scale),
            HsvTarget::Hsl => sse_rgb_to_hsl(r_low_low, g_low_low, b_low_low, v_scale),
        };

        let a_low = _mm_unpacklo_epi8(a_chan, zeros);

        let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
        let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
        let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

        let (x_low_high, y_low_high, z_low_high) = match target {
            HsvTarget::Hsv => sse_rgb_to_hsv(r_low_high, g_low_high, b_low_high, v_scale),
            HsvTarget::Hsl => sse_rgb_to_hsl(r_low_high, g_low_high, b_low_high, v_scale),
        };

        const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
        let x_low = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(x_low_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(x_low_high)),
        );
        let y_low = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(y_low_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(y_low_high)),
        );
        let z_low = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(z_low_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(z_low_high)),
        );

        if USE_ALPHA {
            let ptr = dst_ptr.add(cx * channels);
            store_and_interleave_v4_u16!(ptr, x_low, y_low, z_low, a_low);
        } else {
            let ptr = dst_ptr.add(cx * channels);
            store_and_interleave_v3_u16!(ptr, x_low, y_low, z_low);
        }

        let r_high = _mm_unpackhi_epi8(r_chan, zeros);
        let g_high = _mm_unpackhi_epi8(g_chan, zeros);
        let b_high = _mm_unpackhi_epi8(b_chan, zeros);

        let r_high_low = _mm_unpacklo_epi16(r_high, zeros);
        let g_high_low = _mm_unpacklo_epi16(g_high, zeros);
        let b_high_low = _mm_unpacklo_epi16(b_high, zeros);

        let (x_high_low, y_high_low, z_high_low) = match target {
            HsvTarget::Hsv => sse_rgb_to_hsv(r_high_low, g_high_low, b_high_low, v_scale),
            HsvTarget::Hsl => sse_rgb_to_hsl(r_high_low, g_high_low, b_high_low, v_scale),
        };

        let a_high = _mm_unpackhi_epi8(a_chan, zeros);

        let r_high_high = _mm_unpackhi_epi16(r_high, zeros);
        let g_high_high = _mm_unpackhi_epi16(g_high, zeros);
        let b_high_high = _mm_unpackhi_epi16(b_high, zeros);

        let (x_high_high, y_high_high, z_high_high) = match target {
            HsvTarget::Hsv => sse_rgb_to_hsv(r_high_high, g_high_high, b_high_high, v_scale),
            HsvTarget::Hsl => sse_rgb_to_hsl(r_high_high, g_high_high, b_high_high, v_scale),
        };

        let x_high = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(x_high_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(x_high_high)),
        );
        let y_high = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(y_high_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(y_high_high)),
        );
        let z_high = _mm_packus_epi32(
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(z_high_low)),
            _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(z_high_high)),
        );

        if USE_ALPHA {
            let ptr = dst_ptr.add(cx * channels + 8 * channels);
            store_and_interleave_v4_u16!(ptr, x_high, y_high, z_high, a_high);
        } else {
            let ptr = dst_ptr.add(cx * channels + 8 * channels);
            store_and_interleave_v3_u16!(ptr, x_high, y_high, z_high);
        }

        cx += 16;
    }

    cx
}
