/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

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
#[allow(unused_imports)]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

/// Adds alpha plane into an existing RGB/XYZ/LAB or other 3 plane image. Image will become RGBA, XYZa, LABa etc.
pub fn append_alpha(
    dst: &mut [f32],
    dst_stride: u32,
    src: &[f32],
    src_stride: u32,
    a_plane: &[f32],
    a_stride: u32,
    width: u32,
    height: u32,
) {
    let mut dst_offset = 0usize;
    let mut src_offset = 0usize;
    let mut a_offset = 0usize;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let mut _use_avx = std::arch::is_x86_feature_detected!("avx2");

    for _ in 0..height {
        let mut _cx = 0usize;

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *const f32 };
        let a_ptr = unsafe { (a_plane.as_ptr() as *const u8).add(a_offset) as *const f32 };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            if _use_avx {
                concat_alpha_avx(width, _cx, src_ptr, a_ptr, dst_ptr);
            }
        }

        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        unsafe {
            if _use_sse {
                concat_alpha_sse(width, _cx, src_ptr, a_ptr, dst_ptr);
            }
        }

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            while _cx + 4 < width as usize {
                let xyz_pixel = vld3q_f32(src_ptr.add(_cx * 3usize));
                let a_pixel = vld1q_f32(a_ptr.add(_cx));
                let dst_pixel = float32x4x4_t(xyz_pixel.0, xyz_pixel.1, xyz_pixel.2, a_pixel);
                vst4q_f32(dst_ptr.add(_cx * 4), dst_pixel);
                _cx += 4;
            }
        }

        for x in _cx..width as usize {
            unsafe {
                let px = x * 4;
                let s_x = x * 3;
                let dst = dst_ptr.add(px);
                let src = src_ptr.add(s_x);
                dst.write_unaligned(src.read_unaligned());
                dst.add(1).write_unaligned(src.add(1).read_unaligned());
                dst.add(2).write_unaligned(src.add(2).read_unaligned());
                dst.add(3).write_unaligned(a_ptr.add(x).read_unaligned());
            }
        }

        dst_offset += dst_stride as usize;
        a_offset += a_stride as usize;
        src_offset += src_stride as usize;
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn concat_alpha_sse(
    width: u32,
    mut _cx: usize,
    src_ptr: *const f32,
    a_ptr: *const f32,
    dst_ptr: *mut f32,
) {
    while _cx + 4 < width as usize {
        let xyz_chan_ptr = src_ptr.add(_cx * 3usize);
        let a_chan_ptr = a_ptr.add(_cx);
        let xyz0 = _mm_loadu_ps(xyz_chan_ptr);
        let xyz1 = _mm_loadu_ps(xyz_chan_ptr.add(4));
        let xyz2 = _mm_loadu_ps(xyz_chan_ptr.add(8));
        let a_pixel = _mm_loadu_ps(a_chan_ptr);
        let (x_p, y_p, z_p) = sse_deinterleave_rgb_ps(xyz0, xyz1, xyz2);
        let (xyza0, xyza1, xyza2, xyza3) = sse_interleave_ps_rgba(x_p, y_p, z_p, a_pixel);
        let xyza_chan_ptr = dst_ptr.add(_cx * 4usize);
        _mm_storeu_ps(xyza_chan_ptr, xyza0);
        _mm_storeu_ps(xyza_chan_ptr.add(4), xyza1);
        _mm_storeu_ps(xyza_chan_ptr.add(8), xyza2);
        _mm_storeu_ps(xyza_chan_ptr.add(12), xyza3);
        _cx += 4;
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn concat_alpha_avx(
    width: u32,
    mut _cx: usize,
    src_ptr: *const f32,
    a_ptr: *const f32,
    dst_ptr: *mut f32,
) {
    while _cx + 8 < width as usize {
        let xyz_chan_ptr = src_ptr.add(_cx * 3usize);
        let a_chan_ptr = a_ptr.add(_cx);
        let xyz0 = _mm256_loadu_ps(xyz_chan_ptr);
        let xyz1 = _mm256_loadu_ps(xyz_chan_ptr.add(8));
        let xyz2 = _mm256_loadu_ps(xyz_chan_ptr.add(16));
        let a_pixel = _mm256_loadu_ps(a_chan_ptr);
        let (x_p, y_p, z_p) = avx2_deinterleave_rgb_ps(xyz0, xyz1, xyz2);

        let xyza_chan_ptr = dst_ptr.add(_cx * 4usize);

        let (xyza0, xyza1, xyza2, xyza3) = avx2_interleave_rgba_ps(x_p, y_p, z_p, a_pixel);
        _mm256_store_ps(xyza_chan_ptr, xyza0);
        _mm256_store_ps(xyza_chan_ptr.add(8), xyza1);
        _mm256_store_ps(xyza_chan_ptr.add(16), xyza2);
        _mm256_store_ps(xyza_chan_ptr.add(32), xyza3);
        _cx += 8;
    }
}
