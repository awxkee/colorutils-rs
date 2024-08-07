/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use erydanos::Euclidean3DDistance;

use crate::rgb::Rgb;
use crate::taxicab::TaxicabDistance;
use crate::xyz::Xyz;
use crate::EuclideanDistance;

/// Represents CIELAB color space.
#[derive(Copy, Clone, Debug, Default, PartialOrd, PartialEq)]
pub struct Lab {
    /// `l`: lightness component (0 to 100)
    pub l: f32,
    /// `a`: green (negative) and red (positive) component.
    pub a: f32,
    /// `b`: blue (negative) and yellow (positive) component
    pub b: f32,
}

impl Lab {
    /// Create a new CIELAB color.
    ///
    /// `l`: lightness component (0 to 100)
    ///
    /// `a`: green (negative) and red (positive) component.
    ///
    /// `b`: blue (negative) and yellow (positive) component.
    #[inline]
    pub fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }
}

impl Lab {
    /// Converts to CIE Lab from Rgb
    #[inline]
    pub fn from_rgb(rgb: Rgb<u8>) -> Self {
        let xyz = Xyz::from_srgb(rgb);
        let x = xyz.x * 100f32 / 95.047f32;
        let y = xyz.y * 100f32 / 100f32;
        let z = xyz.z * 100f32 / 108.883f32;
        let x = if x > 0.008856f32 {
            x.cbrt()
        } else {
            7.787f32 * x + 16f32 / 116f32
        };
        let y = if y > 0.008856f32 {
            y.cbrt()
        } else {
            7.787f32 * y + 16f32 / 116f32
        };
        let z = if z > 0.008856f32 {
            z.cbrt()
        } else {
            7.787f32 * z + 16f32 / 116f32
        };
        Self::new((116f32 * y) - 16f32, 500f32 * (x - y), 200f32 * (y - z))
    }
}

impl Lab {
    /// Converts CIE Lab into Rgb
    #[inline]
    pub fn to_rgb8(&self) -> Rgb<u8> {
        let y = (self.l + 16.0) / 116.0;
        let x = self.a * (1f32 / 500f32) + y;
        let z = y - self.b * (1f32 / 200f32);
        let dx = x * x;
        let x3 = dx * x;
        let dy = y * y;
        let y3 = dy * y;
        let dz = z * z;
        let z3 = dz * z;
        let x = 95.047
            * if x3 > 0.008856 {
                x3
            } else {
                (x - 16.0 / 116.0) / 7.787
            };
        let y = 100.0
            * if y3 > 0.008856 {
                y3
            } else {
                (y - 16.0 / 116.0) / 7.787
            };
        let z = 108.883
            * if z3 > 0.008856 {
                z3
            } else {
                (z - 16.0 / 116.0) / 7.787
            };
        Xyz::new(x / 100f32, y / 100f32, z / 100f32).to_srgb()
    }

    /// Converts CIE Lab into Rgb
    #[inline]
    pub fn to_rgb(&self) -> Rgb<u8> {
        self.to_rgb8()
    }
}

impl EuclideanDistance for Lab {
    #[inline]
    fn euclidean_distance(&self, other: Lab) -> f32 {
        (self.l - other.l).hypot3(self.a - other.a, self.b - other.b)
    }
}

impl TaxicabDistance for Lab {
    #[inline]
    fn taxicab_distance(&self, other: Self) -> f32 {
        (self.a - other.a).hypot(self.b - other.b) + (self.l - other.l).abs()
    }
}
