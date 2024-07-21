/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::euclidean::EuclideanDistance;
use crate::hsv::Hsv;
use crate::lab::Lab;
use crate::luv::Luv;
use crate::{Hsl, Jzazbz, LCh, Sigmoidal, TransferFunction, Xyz};
use erydanos::Euclidean3DDistance;

#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
/// Represents any RGB values, Rgb<u8>, Rgb<u16> etc.
pub struct Rgb<T> {
    /// Red component
    pub r: T,
    /// Green component
    pub g: T,
    /// Blue component
    pub b: T,
}

impl Rgb<u8> {
    /// Converts rgb to Jzazbz
    #[inline]
    pub fn to_jzazbz(&self, transfer_function: TransferFunction) -> Jzazbz {
        Jzazbz::from_rgb(*self, transfer_function)
    }

    /// Converts rgb to XYZ
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert RGB into linear RGB
    #[inline]
    pub fn to_xyz(&self, matrix: &[[f32; 3]; 3], transfer_function: TransferFunction) -> Xyz {
        Xyz::from_rgb(*self, matrix, transfer_function)
    }

    /// Converts rgb to HSL
    #[inline]
    pub fn to_hsl(&self) -> Hsl {
        Hsl::from_rgb(*self)
    }

    /// Converts rgb to HSV
    #[inline]
    pub fn to_hsv(&self) -> Hsv {
        Hsv::from(*self)
    }

    /// Converts rgb to CIELAB
    #[inline]
    pub fn to_lab(&self) -> Lab {
        Lab::from_rgb(*self)
    }

    /// Converts rgb to CIELUV
    #[inline]
    pub fn to_luv(&self) -> Luv {
        Luv::from_rgb(*self)
    }

    /// Converts rgb to CIELCH
    #[inline]
    pub fn to_lch(&self) -> LCh {
        LCh::from_rgb(*self)
    }

    /// Converts rgb to RGB f32
    #[inline]
    pub fn to_rgb_f32(&self) -> Rgb<f32> {
        const SCALE: f32 = 1f32 / 255f32;
        Rgb::<f32>::new(
            self.r as f32 * SCALE,
            self.g as f32 * SCALE,
            self.b as f32 * SCALE,
        )
    }

    /// Converts rgb to S-shaped sigmoidized components
    #[inline]
    pub fn to_sigmoidal(&self) -> Sigmoidal {
        Sigmoidal::from_rgb(*self)
    }
}

impl From<Rgb<f32>> for Rgb<u8> {
    #[inline]
    fn from(value: Rgb<f32>) -> Self {
        value.to_u8()
    }
}

impl Rgb<f32> {
    #[inline]
    pub fn apply(&self, gen: fn(f32) -> f32) -> Self {
        Self {
            r: gen(self.r),
            g: gen(self.g),
            b: gen(self.b),
        }
    }

    #[inline]
    pub fn to_u8(&self) -> Rgb<u8> {
        Rgb::<u8>::new(
            (self.r * 255f32).max(0f32).round().min(255f32) as u8,
            (self.g * 255f32).max(0f32).round().min(255f32) as u8,
            (self.b * 255f32).max(0f32).round().min(255f32) as u8,
        )
    }
}

impl From<Sigmoidal> for Rgb<u8> {
    #[inline]
    fn from(value: Sigmoidal) -> Self {
        value.to_rgb()
    }
}

impl<T> Rgb<T> {
    pub fn new(r: T, g: T, b: T) -> Rgb<T> {
        Rgb { r, g, b }
    }
}

impl EuclideanDistance for Rgb<u8> {
    fn euclidean_distance(&self, other: Rgb<u8>) -> f32 {
        (self.r as f32 - other.r as f32).hypot3(
            self.g as f32 - other.g as f32,
            self.b as f32 - other.b as f32,
        )
    }
}

impl EuclideanDistance for Rgb<f32> {
    fn euclidean_distance(&self, other: Rgb<f32>) -> f32 {
        (self.r - other.r).hypot3(self.g - other.g, self.b - other.b)
    }
}

impl EuclideanDistance for Rgb<u16> {
    fn euclidean_distance(&self, other: Rgb<u16>) -> f32 {
        (self.r as f32 - other.r as f32).hypot3(
            self.g as f32 - other.g as f32,
            self.b as f32 - other.b as f32,
        )
    }
}
