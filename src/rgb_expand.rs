#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::*;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Expands RGB to RGBA.
pub fn rgb_to_rgba(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    default_alpha: u8,
) {
    let mut dst_offset = 0usize;
    let mut src_offset = 0usize;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let mut _use_sse = false;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let mut _use_avx = false;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        #[cfg(target_feature = "sse4.1")]
        if is_x86_feature_detected!("sse4.1") {
            _use_sse = true;
        }
        #[cfg(target_feature = "avx2")]
        if is_x86_feature_detected!("avx2") {
            _use_avx = true;
        }
    }

    for _ in 0..height as usize {
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            let src_ptr = src.as_ptr().add(src_offset);
            let dst_ptr = dst.as_mut_ptr().add(dst_offset);
            if _use_avx {
                let v_alpha = _mm256_set1_epi8(default_alpha as i8);
                while cx + 32 < width as usize {
                    let xyz_chan_ptr = src_ptr.add(cx * 3usize);
                    let xyz0 = _mm256_loadu_si256(xyz_chan_ptr as *const __m256i);
                    let xyz1 = _mm256_loadu_si256(xyz_chan_ptr.add(32) as *const __m256i);
                    let xyz2 = _mm256_loadu_si256(xyz_chan_ptr.add(64) as *const __m256i);
                    let (x_p, y_p, z_p) = avx2_deinterleave_rgb(xyz0, xyz1, xyz2);

                    let xyza_chan_ptr = dst_ptr.add(cx * 4usize);

                    let (xyza0, xyza1, xyza2, xyza3) =
                        avx2_interleave_rgba_epi8(x_p, y_p, z_p, v_alpha);
                    _mm256_storeu_si256(xyza_chan_ptr as *mut __m256i, xyza0);
                    _mm256_storeu_si256(xyza_chan_ptr.add(32) as *mut __m256i, xyza1);
                    _mm256_storeu_si256(xyza_chan_ptr.add(64) as *mut __m256i, xyza2);
                    _mm256_storeu_si256(xyza_chan_ptr.add(96) as *mut __m256i, xyza3);
                    cx += 32;
                }
            }
            if _use_sse {
                let v_alpha = _mm_set1_epi8(default_alpha as i8);
                while cx + 16 < width as usize {
                    let xyz_chan_ptr = src_ptr.add(cx * 3usize);
                    let xyz0 = _mm_loadu_si128(xyz_chan_ptr as *const __m128i);
                    let xyz1 = _mm_loadu_si128(xyz_chan_ptr.add(16) as *const __m128i);
                    let xyz2 = _mm_loadu_si128(xyz_chan_ptr.add(32) as *const __m128i);
                    let (x_p, y_p, z_p) = sse_deinterleave_rgb(xyz0, xyz1, xyz2);
                    let (xyza0, xyza1, xyza2, xyza3) = sse_interleave_rgba(x_p, y_p, z_p, v_alpha);
                    let xyza_chan_ptr = dst_ptr.add(cx * 4usize);
                    _mm_storeu_si128(xyza_chan_ptr as *mut __m128i, xyza0);
                    _mm_storeu_si128(xyza_chan_ptr.add(16) as *mut __m128i, xyza1);
                    _mm_storeu_si128(xyza_chan_ptr.add(32) as *mut __m128i, xyza2);
                    _mm_storeu_si128(xyza_chan_ptr.add(48) as *mut __m128i, xyza3);
                    cx += 16;
                }
            }
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            let v_alpha = vdupq_n_u8(default_alpha);
            let src_ptr = src.as_ptr().add(src_offset);
            let dst_ptr = dst.as_mut_ptr().add(dst_offset);
            while cx + 16 < width as usize {
                let xyz_pixel = vld3q_u8(src_ptr.add(cx * 3usize));
                let dst_pixel = uint8x16x4_t(xyz_pixel.0, xyz_pixel.1, xyz_pixel.2, v_alpha);
                vst4q_u8(dst_ptr.add(cx * 4), dst_pixel);
                cx += 16;
            }
        }

        for x in cx..width as usize {
            let px = dst_offset + x * 4;
            let s_x = src_offset + x * 3;
            unsafe {
                *dst.get_unchecked_mut(px) = *src.get_unchecked(s_x);
                *dst.get_unchecked_mut(px + 1) = *src.get_unchecked(s_x + 1);
                *dst.get_unchecked_mut(px + 2) = *src.get_unchecked(s_x + 2);
                *dst.get_unchecked_mut(px + 3) = default_alpha;
            }
        }

        dst_offset += dst_stride as usize;
        src_offset += src_stride as usize;
    }
}

/// Expands BGR to BGRA.
pub fn bgr_to_bgra(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    default_alpha: u8,
) {
    rgb_to_rgba(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        default_alpha,
    );
}
