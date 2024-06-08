/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2_to_xyz_lab;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2_utils;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx_gamma_curves;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx_math;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx_support;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx_color;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx_xyza_laba_to_image;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx_xyz_lab_to_image;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use avx2_to_xyz_lab::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use avx2_utils::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use avx_math::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use avx_support::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use avx_xyza_laba_to_image::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use avx_xyz_lab_to_image::*;