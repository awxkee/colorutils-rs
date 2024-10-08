/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod sse_image_to_linear_unsigned {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use crate::image::ImageConfiguration;
    use crate::sse::*;
    use crate::{
        load_u8_and_deinterleave, store_and_interleave_v3_half_u8, store_and_interleave_v3_u8,
        store_and_interleave_v4_half_u8, store_and_interleave_v4_u8, TransferFunction,
    };

    #[inline(always)]
    unsafe fn sse_triple_to_linear_u8<const INTO_LINEAR: bool>(
        r: __m128i,
        g: __m128i,
        b: __m128i,
        transfer_function: TransferFunction,
    ) -> (__m128i, __m128i, __m128i) {
        let u8_scale = _mm_set1_ps(1f32 / 255f32);
        let r_f = _mm_mul_ps(_mm_cvtepi32_ps(r), u8_scale);
        let g_f = _mm_mul_ps(_mm_cvtepi32_ps(g), u8_scale);
        let b_f = _mm_mul_ps(_mm_cvtepi32_ps(b), u8_scale);
        let u8_backwards = _mm_set1_ps(255f32);
        let r_linear = _mm_mul_ps(
            match INTO_LINEAR {
                true => perform_sse_linear_transfer(transfer_function, r_f),
                false => perform_sse_gamma_transfer(transfer_function, r_f),
            },
            u8_backwards,
        );
        let g_linear = _mm_mul_ps(
            match INTO_LINEAR {
                true => perform_sse_linear_transfer(transfer_function, g_f),
                false => perform_sse_gamma_transfer(transfer_function, g_f),
            },
            u8_backwards,
        );
        let b_linear = _mm_mul_ps(
            match INTO_LINEAR {
                true => perform_sse_linear_transfer(transfer_function, b_f),
                false => perform_sse_gamma_transfer(transfer_function, b_f),
            },
            u8_backwards,
        );
        (
            _mm_cvtps_epi32(r_linear),
            _mm_cvtps_epi32(g_linear),
            _mm_cvtps_epi32(b_linear),
        )
    }

    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn sse_channels_to_linear_u8<
        const CHANNELS_CONFIGURATION: u8,
        const USE_ALPHA: bool,
        const INTO_LINEAR: bool,
    >(
        start_cx: usize,
        src: *const u8,
        src_offset: usize,
        width: u32,
        dst: *mut u8,
        dst_offset: usize,
        transfer_function: TransferFunction,
    ) -> usize {
        let image_configuration: ImageConfiguration = CHANNELS_CONFIGURATION.into();
        let channels = image_configuration.get_channels_count();
        let mut cx = start_cx;

        let dst_ptr = dst.add(dst_offset);

        let zeros = _mm_setzero_si128();

        while cx + 16 < width as usize {
            let src_ptr = src.add(src_offset + cx * channels);
            let (r_chan, g_chan, b_chan, a_chan) =
                load_u8_and_deinterleave!(src_ptr, image_configuration);

            let r_low = _mm_cvtepu8_epi16(r_chan);
            let g_low = _mm_cvtepu8_epi16(g_chan);
            let b_low = _mm_cvtepu8_epi16(b_chan);

            let r_low_low = _mm_cvtepu16_epi32(r_low);
            let g_low_low = _mm_cvtepu16_epi32(g_low);
            let b_low_low = _mm_cvtepu16_epi32(b_low);

            let (x_low_low, y_low_low, z_low_low) = sse_triple_to_linear_u8::<INTO_LINEAR>(
                r_low_low,
                g_low_low,
                b_low_low,
                transfer_function,
            );

            let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
            let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
            let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

            let (x_low_high, y_low_high, z_low_high) = sse_triple_to_linear_u8::<INTO_LINEAR>(
                r_low_high,
                g_low_high,
                b_low_high,
                transfer_function,
            );

            let r_high = _mm_unpackhi_epi8(r_chan, zeros);
            let g_high = _mm_unpackhi_epi8(g_chan, zeros);
            let b_high = _mm_unpackhi_epi8(b_chan, zeros);

            let r_high_low = _mm_cvtepu16_epi32(r_high);
            let g_high_low = _mm_cvtepu16_epi32(g_high);
            let b_high_low = _mm_cvtepu16_epi32(b_high);

            let (x_high_low, y_high_low, z_high_low) = sse_triple_to_linear_u8::<INTO_LINEAR>(
                r_high_low,
                g_high_low,
                b_high_low,
                transfer_function,
            );

            let r_high_high = _mm_unpackhi_epi16(r_high, zeros);
            let g_high_high = _mm_unpackhi_epi16(g_high, zeros);
            let b_high_high = _mm_unpackhi_epi16(b_high, zeros);

            let (x_high_high, y_high_high, z_high_high) = sse_triple_to_linear_u8::<INTO_LINEAR>(
                r_high_high,
                g_high_high,
                b_high_high,
                transfer_function,
            );

            let r_u_norm = _mm_packus_epi16(
                _mm_packus_epi32(x_low_low, x_low_high),
                _mm_packus_epi32(x_high_low, x_high_high),
            );

            let g_u_norm = _mm_packus_epi16(
                _mm_packus_epi32(y_low_low, y_low_high),
                _mm_packus_epi32(y_high_low, y_high_high),
            );

            let b_u_norm = _mm_packus_epi16(
                _mm_packus_epi32(z_low_low, z_low_high),
                _mm_packus_epi32(z_high_low, z_high_high),
            );

            let dst = dst_ptr.add(cx * channels);

            if USE_ALPHA {
                store_and_interleave_v4_u8!(
                    dst,
                    image_configuration,
                    r_u_norm,
                    g_u_norm,
                    b_u_norm,
                    a_chan
                );
            } else {
                store_and_interleave_v3_u8!(dst, image_configuration, r_u_norm, g_u_norm, b_u_norm);
            }

            cx += 16;
        }

        while cx + 8 < width as usize {
            let src_ptr = src.add(src_offset + cx * channels);
            let (r_chan, g_chan, b_chan, a_chan) =
                load_u8_and_deinterleave!(src_ptr, image_configuration);

            let r_low = _mm_cvtepu8_epi16(r_chan);
            let g_low = _mm_cvtepu8_epi16(g_chan);
            let b_low = _mm_cvtepu8_epi16(b_chan);

            let r_low_low = _mm_cvtepu16_epi32(r_low);
            let g_low_low = _mm_cvtepu16_epi32(g_low);
            let b_low_low = _mm_cvtepu16_epi32(b_low);

            let (x_low_low, y_low_low, z_low_low) = sse_triple_to_linear_u8::<INTO_LINEAR>(
                r_low_low,
                g_low_low,
                b_low_low,
                transfer_function,
            );

            let r_low_high = _mm_unpackhi_epi16(r_low, zeros);
            let g_low_high = _mm_unpackhi_epi16(g_low, zeros);
            let b_low_high = _mm_unpackhi_epi16(b_low, zeros);

            let (x_low_high, y_low_high, z_low_high) = sse_triple_to_linear_u8::<INTO_LINEAR>(
                r_low_high,
                g_low_high,
                b_low_high,
                transfer_function,
            );

            let r_u_norm = _mm_packus_epi16(_mm_packus_epi32(x_low_low, x_low_high), zeros);

            let g_u_norm = _mm_packus_epi16(_mm_packus_epi32(y_low_low, y_low_high), zeros);

            let b_u_norm = _mm_packus_epi16(_mm_packus_epi32(z_low_low, z_low_high), zeros);

            let dst = dst_ptr.add(cx * channels);

            if USE_ALPHA {
                store_and_interleave_v4_half_u8!(
                    dst,
                    image_configuration,
                    r_u_norm,
                    g_u_norm,
                    b_u_norm,
                    a_chan
                );
            } else {
                store_and_interleave_v3_half_u8!(
                    dst,
                    image_configuration,
                    r_u_norm,
                    g_u_norm,
                    b_u_norm
                );
            }

            cx += 8;
        }

        cx
    }
}
