/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
mod avx;
mod concat_alpha;
mod euclidean;
mod gamma_curves;
mod hsl;
mod hsv;
mod hsv_to_image;
mod image;
mod image_to_hsv;
mod image_to_hsv_support;
mod image_to_jzazbz;
mod image_to_linear;
mod image_to_linear_u8;
mod image_to_oklab;
mod image_to_sigmoidal;
mod image_to_xyz_lab;
mod image_xyza_laba;
mod jzazbz;
mod jzazbz_to_image;
mod jzczhz;
mod lab;
mod linear_to_image;
mod linear_to_image_u8;
pub mod linear_to_planar;
mod luv;
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
mod neon;
mod oklab;
mod oklab_to_image;
mod oklch;
pub mod planar_to_linear;
mod rgb;
mod rgb_expand;
mod rgba;
mod sigmoidal;
mod sigmoidal_to_image;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
mod sse;
mod taxicab;
mod xyz;
mod xyz_lab_to_image;
mod xyz_target;
mod xyz_transform;
mod xyza_laba_to_image;

pub use concat_alpha::append_alpha;
pub use gamma_curves::*;
pub use hsl::Hsl;
pub use hsv::Hsv;
pub use hsv_to_image::*;
pub use image_to_hsv::*;
pub use image_to_linear::*;
pub use image_to_linear_u8::*;
pub use image_to_xyz_lab::bgr_to_lab;
pub use image_to_xyz_lab::bgr_to_lch;
pub use image_to_xyz_lab::bgr_to_luv;
pub use image_to_xyz_lab::bgra_to_laba;
pub use image_to_xyz_lab::rgb_to_lab;
pub use image_to_xyz_lab::rgb_to_lch;
pub use image_to_xyz_lab::rgb_to_luv;
pub use image_to_xyz_lab::rgb_to_xyz;
pub use image_to_xyz_lab::rgba_to_lab;
pub use image_to_xyz_lab::rgba_to_laba;
pub use image_to_xyz_lab::rgba_to_xyz;
pub use image_to_xyz_lab::rgba_to_xyza;
pub use image_to_xyz_lab::srgb_to_xyz;
pub use image_to_xyz_lab::srgba_to_xyz;
pub use image_to_xyz_lab::srgba_to_xyza;
pub use image_xyza_laba::bgra_to_lab_with_alpha;
pub use image_xyza_laba::bgra_to_lch_with_alpha;
pub use image_xyza_laba::bgra_to_luv_with_alpha;
pub use image_xyza_laba::bgra_to_xyz_with_alpha;
pub use image_xyza_laba::rgba_to_lab_with_alpha;
pub use image_xyza_laba::rgba_to_lch_with_alpha;
pub use image_xyza_laba::rgba_to_luv_with_alpha;
pub use image_xyza_laba::rgba_to_xyz_with_alpha;
pub use lab::Lab;
pub use linear_to_image::*;
pub use linear_to_image_u8::*;
pub use luv::LCh;
pub use luv::Luv;
pub use rgb::Rgb;
pub use rgba::Rgb565;
pub use rgba::Rgba;
pub use rgba::Rgba1010102;
pub use rgba::ToRgb565;
pub use rgba::ToRgba1010102;
pub use rgba::ToRgba8;
pub use rgba::ToRgbaF16;
pub use rgba::ToRgbaF32;
pub use xyz::Xyz;
pub use xyz_lab_to_image::lab_to_srgb;
pub use xyz_lab_to_image::laba_to_srgb;
pub use xyz_lab_to_image::lch_to_bgr;
pub use xyz_lab_to_image::lch_to_rgb;
pub use xyz_lab_to_image::luv_to_bgr;
pub use xyz_lab_to_image::luv_to_rgb;
pub use xyz_lab_to_image::xyz_to_rgb;
pub use xyz_lab_to_image::xyz_to_srgb;
pub use xyz_lab_to_image::xyza_to_rgba;
pub use xyz_transform::*;
pub use xyza_laba_to_image::lab_with_alpha_to_bgra;
pub use xyza_laba_to_image::lab_with_alpha_to_rgba;
pub use xyza_laba_to_image::lch_with_alpha_to_bgra;
pub use xyza_laba_to_image::lch_with_alpha_to_rgba;
pub use xyza_laba_to_image::luv_with_alpha_to_bgra;
pub use xyza_laba_to_image::luv_with_alpha_to_rgba;
pub use xyza_laba_to_image::xyz_with_alpha_to_bgra;
pub use xyza_laba_to_image::xyz_with_alpha_to_rgba;

pub use euclidean::EuclideanDistance;
pub use image_to_jzazbz::bgr_to_jzazbz;
pub use image_to_jzazbz::bgr_to_jzczhz;
pub use image_to_jzazbz::bgra_to_jzazbz;
pub use image_to_jzazbz::bgra_to_jzczhz;
pub use image_to_jzazbz::rgb_to_jzazbz;
pub use image_to_jzazbz::rgb_to_jzczhz;
pub use image_to_jzazbz::rgba_to_jzazbz;
pub use image_to_jzazbz::rgba_to_jzczhz;
pub use image_to_oklab::bgr_to_oklab;
pub use image_to_oklab::bgr_to_oklch;
pub use image_to_oklab::bgra_to_oklab;
pub use image_to_oklab::bgra_to_oklch;
pub use image_to_oklab::rgb_to_oklab;
pub use image_to_oklab::rgb_to_oklch;
pub use image_to_oklab::rgba_to_oklab;
pub use image_to_oklab::rgba_to_oklch;
pub use image_to_sigmoidal::bgra_to_sigmoidal;
pub use image_to_sigmoidal::rgb_to_sigmoidal;
pub use image_to_sigmoidal::rgba_to_sigmoidal;
pub use jzazbz::Jzazbz;
pub use jzazbz_to_image::jzazbz_to_bgr;
pub use jzazbz_to_image::jzazbz_to_bgra;
pub use jzazbz_to_image::jzazbz_to_rgb;
pub use jzazbz_to_image::jzazbz_to_rgba;
pub use jzazbz_to_image::jzczhz_to_bgr;
pub use jzazbz_to_image::jzczhz_to_bgra;
pub use jzazbz_to_image::jzczhz_to_rgb;
pub use jzazbz_to_image::jzczhz_to_rgba;
pub use jzczhz::Jzczhz;
pub use oklab::Oklab;
pub use oklab_to_image::oklab_to_bgr;
pub use oklab_to_image::oklab_to_bgra;
pub use oklab_to_image::oklab_to_rgb;
pub use oklab_to_image::oklab_to_rgba;
pub use oklab_to_image::oklch_to_bgr;
pub use oklab_to_image::oklch_to_bgra;
pub use oklab_to_image::oklch_to_rgb;
pub use oklab_to_image::oklch_to_rgba;
pub use oklch::Oklch;
pub use rgb_expand::*;
pub use sigmoidal::Sigmoidal;
pub use sigmoidal_to_image::sigmoidal_to_bgra;
pub use sigmoidal_to_image::sigmoidal_to_rgb;
pub use sigmoidal_to_image::sigmoidal_to_rgba;
pub use taxicab::TaxicabDistance;
