/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::rgb::Rgb;
use crate::taxicab::TaxicabDistance;
use crate::xyz::Xyz;
use crate::EuclideanDistance;
use num_traits::Pow;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

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
    /// Converts to CIE Lab from CIE XYZ
    #[inline]
    pub fn from_xyz(xyz: Xyz) -> Self {
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

    /// Converts to CIE Lab from Rgb
    #[inline]
    pub fn from_rgb(rgb: Rgb<u8>) -> Self {
        let xyz = Xyz::from_srgb(rgb);
        Self::from_xyz(xyz)
    }

    pub const fn l_range() -> (f32, f32) {
        (0., 100.)
    }

    pub const fn a_range() -> (f32, f32) {
        (-86.185f32, 98.254f32)
    }

    pub const fn b_range() -> (f32, f32) {
        (-107.863f32, 94.482f32)
    }
}

impl Lab {
    /// Converts CIE Lab into CIE XYZ
    #[inline]
    pub fn to_xyz(&self) -> Xyz {
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
        Xyz::new(x * (1. / 100f32), y * (1. / 100f32), z * (1. / 100f32))
    }

    /// Converts CIE Lab into Rgb
    #[inline]
    pub fn to_rgb8(&self) -> Rgb<u8> {
        let xyz = self.to_xyz();
        Xyz::new(xyz.x, xyz.y, xyz.z).to_srgb()
    }

    /// Converts CIE Lab into Rgb
    #[inline]
    pub fn to_rgb(&self) -> Rgb<u8> {
        self.to_rgb8()
    }

    #[inline]
    pub fn hybrid_distance(&self, other: Self) -> f32 {
        let lax = self.l - other.l;
        let dax = self.a - other.a;
        let bax = self.b - other.b;
        (dax * dax + bax * bax).sqrt() + lax.abs()
    }
}

impl EuclideanDistance for Lab {
    #[inline]
    fn euclidean_distance(&self, other: Lab) -> f32 {
        let lax = self.l - other.l;
        let dax = self.a - other.a;
        let bax = self.b - other.b;
        (lax * lax + dax * dax + bax * bax).sqrt()
    }
}

impl TaxicabDistance for Lab {
    #[inline]
    fn taxicab_distance(&self, other: Self) -> f32 {
        let lax = self.l - other.l;
        let dax = self.a - other.a;
        let bax = self.b - other.b;
        lax.abs() + dax.abs() + bax.abs()
    }
}

impl Index<usize> for Lab {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.l,
            1 => &self.a,
            2 => &self.b,
            _ => panic!("Index out of bounds for Lab"),
        }
    }
}

impl IndexMut<usize> for Lab {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.l,
            1 => &mut self.a,
            2 => &mut self.b,
            _ => panic!("Index out of bounds for Lab"),
        }
    }
}

impl Add<Lab> for Lab {
    type Output = Lab;

    #[inline]
    fn add(self, rhs: Self) -> Lab {
        Lab::new(self.l + rhs.l, self.a + rhs.a, self.b + rhs.b)
    }
}

impl Add<f32> for Lab {
    type Output = Lab;

    #[inline]
    fn add(self, rhs: f32) -> Lab {
        Lab::new(self.l + rhs, self.a + rhs, self.b + rhs)
    }
}

impl AddAssign<Lab> for Lab {
    #[inline]
    fn add_assign(&mut self, rhs: Lab) {
        self.l += rhs.l;
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl AddAssign<f32> for Lab {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.l += rhs;
        self.a += rhs;
        self.b += rhs;
    }
}

impl Mul<f32> for Lab {
    type Output = Lab;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Lab::new(self.l * rhs, self.a * rhs, self.b * rhs)
    }
}

impl Mul<Lab> for Lab {
    type Output = Lab;

    #[inline]
    fn mul(self, rhs: Lab) -> Self::Output {
        Lab::new(self.l * rhs.l, self.a * rhs.a, self.b * rhs.b)
    }
}

impl MulAssign<f32> for Lab {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.l *= rhs;
        self.a *= rhs;
        self.b *= rhs;
    }
}

impl MulAssign<Lab> for Lab {
    #[inline]
    fn mul_assign(&mut self, rhs: Lab) {
        self.l *= rhs.l;
        self.a *= rhs.a;
        self.b *= rhs.b;
    }
}

impl Sub<f32> for Lab {
    type Output = Lab;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Lab::new(self.l - rhs, self.a - rhs, self.b - rhs)
    }
}

impl Sub<Lab> for Lab {
    type Output = Lab;

    #[inline]
    fn sub(self, rhs: Lab) -> Self::Output {
        Lab::new(self.l - rhs.l, self.a - rhs.a, self.b - rhs.b)
    }
}

impl SubAssign<f32> for Lab {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.l -= rhs;
        self.a -= rhs;
        self.b -= rhs;
    }
}

impl SubAssign<Lab> for Lab {
    #[inline]
    fn sub_assign(&mut self, rhs: Lab) {
        self.l -= rhs.l;
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl Div<f32> for Lab {
    type Output = Lab;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Lab::new(self.l / rhs, self.a / rhs, self.b / rhs)
    }
}

impl Div<Lab> for Lab {
    type Output = Lab;

    #[inline]
    fn div(self, rhs: Lab) -> Self::Output {
        Lab::new(self.l / rhs.l, self.a / rhs.a, self.b / rhs.b)
    }
}

impl DivAssign<f32> for Lab {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.l /= rhs;
        self.a /= rhs;
        self.b /= rhs;
    }
}

impl DivAssign<Lab> for Lab {
    #[inline]
    fn div_assign(&mut self, rhs: Lab) {
        self.l /= rhs.l;
        self.a /= rhs.a;
        self.b /= rhs.b;
    }
}

impl Neg for Lab {
    type Output = Lab;

    #[inline]
    fn neg(self) -> Self::Output {
        Lab::new(-self.l, -self.a, -self.b)
    }
}

impl Pow<f32> for Lab {
    type Output = Lab;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Lab::new(self.l * rhs, self.a * rhs, self.b * rhs)
    }
}

impl Pow<Lab> for Lab {
    type Output = Lab;

    #[inline]
    fn pow(self, rhs: Lab) -> Self::Output {
        Lab::new(self.l.powf(rhs.l), self.a.powf(rhs.a), self.b.powf(self.b))
    }
}

impl Lab {
    #[inline]
    pub fn sqrt(&self) -> Lab {
        Lab::new(
            if self.l < 0. { 0. } else { self.l.sqrt() },
            if self.a < 0. { 0. } else { self.a.sqrt() },
            if self.b < 0. { 0. } else { self.b.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Lab {
        Lab::new(self.l.cbrt(), self.a.cbrt(), self.b.cbrt())
    }
}
