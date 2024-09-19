/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{EuclideanDistance, Oklab, Rgb, TaxicabDistance, TransferFunction};
use erydanos::{eatan2f, ehypotf, Cosine, Sine};
use num_traits::{Float, Pow};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Represents *Oklch* colorspace
#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub struct Oklch {
    /// Lightness
    pub l: f32,
    /// Chroma
    pub c: f32,
    /// Hue
    pub h: f32,
}

impl Oklch {
    /// Creates new instance
    #[inline]
    pub fn new(l: f32, c: f32, h: f32) -> Oklch {
        Oklch { l, c, h }
    }

    /// Converts *Rgb* into *Oklch*
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function into linear colorspace and its inverse
    #[inline]
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> Oklch {
        let oklab = rgb.to_oklab(transfer_function);
        Oklch::from_oklab(oklab)
    }

    /// Converts *Oklch* into *Rgb*
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function into linear colorspace and its inverse
    #[inline]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let oklab = self.to_oklab();
        oklab.to_rgb(transfer_function)
    }

    /// Converts *Oklab* to *Oklch*
    #[inline]
    pub fn from_oklab(oklab: Oklab) -> Oklch {
        let chroma = ehypotf(oklab.b, oklab.a);
        let hue = eatan2f(oklab.b, oklab.a);
        Oklch::new(oklab.l, chroma, hue)
    }

    /// Converts *Oklch* to *Oklab*
    #[inline]
    pub fn to_oklab(&self) -> Oklab {
        let l = self.l;
        let a = self.c * self.h.ecos();
        let b = self.c * self.h.esin();
        Oklab::new(l, a, b)
    }
}

impl EuclideanDistance for Oklch {
    #[inline]
    fn euclidean_distance(&self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let dc = self.c - other.c;
        let dh = self.h - other.h;
        (dl * dl + dc * dc + dh * dh).sqrt()
    }
}

impl TaxicabDistance for Oklch {
    #[inline]
    fn taxicab_distance(&self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let dc = self.c - other.c;
        let dh = self.h - other.h;
        dl.abs() + dc.abs() + dh.abs()
    }
}

impl Add<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn add(self, rhs: Self) -> Oklch {
        Oklch::new(self.l + rhs.l, self.c + rhs.c, self.h + rhs.h)
    }
}

impl Add<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn add(self, rhs: f32) -> Oklch {
        Oklch::new(self.l + rhs, self.c + rhs, self.h + rhs)
    }
}

impl AddAssign<Oklch> for Oklch {
    #[inline]
    fn add_assign(&mut self, rhs: Oklch) {
        self.l += rhs.l;
        self.c += rhs.c;
        self.h += rhs.h;
    }
}

impl AddAssign<f32> for Oklch {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.l += rhs;
        self.c += rhs;
        self.h += rhs;
    }
}

impl Mul<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Oklch::new(self.l * rhs, self.c * rhs, self.h * rhs)
    }
}

impl Mul<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn mul(self, rhs: Oklch) -> Self::Output {
        Oklch::new(self.l * rhs.l, self.c * rhs.c, self.h * rhs.h)
    }
}

impl MulAssign<f32> for Oklch {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.l *= rhs;
        self.c *= rhs;
        self.h *= rhs;
    }
}

impl MulAssign<Oklch> for Oklch {
    #[inline]
    fn mul_assign(&mut self, rhs: Oklch) {
        self.l *= rhs.l;
        self.c *= rhs.c;
        self.h *= rhs.h;
    }
}

impl Sub<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Oklch::new(self.l - rhs, self.c - rhs, self.h - rhs)
    }
}

impl Sub<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn sub(self, rhs: Oklch) -> Self::Output {
        Oklch::new(self.l - rhs.l, self.c - rhs.c, self.h - rhs.h)
    }
}

impl SubAssign<f32> for Oklch {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.l -= rhs;
        self.c -= rhs;
        self.h -= rhs;
    }
}

impl SubAssign<Oklch> for Oklch {
    #[inline]
    fn sub_assign(&mut self, rhs: Oklch) {
        self.l -= rhs.l;
        self.c -= rhs.c;
        self.h -= rhs.h;
    }
}

impl Div<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Oklch::new(self.l / rhs, self.c / rhs, self.h / rhs)
    }
}

impl Div<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn div(self, rhs: Oklch) -> Self::Output {
        Oklch::new(self.l / rhs.l, self.c / rhs.c, self.h / rhs.h)
    }
}

impl DivAssign<f32> for Oklch {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.l /= rhs;
        self.c /= rhs;
        self.h /= rhs;
    }
}

impl DivAssign<Oklch> for Oklch {
    #[inline]
    fn div_assign(&mut self, rhs: Oklch) {
        self.l /= rhs.l;
        self.c /= rhs.c;
        self.h /= rhs.h;
    }
}

impl Neg for Oklch {
    type Output = Oklch;

    fn neg(self) -> Self::Output {
        Oklch::new(-self.l, -self.c, -self.h)
    }
}

impl Pow<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Oklch::new(self.l.powf(rhs), self.c.powf(rhs), self.h.powf(rhs))
    }
}

impl Pow<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn pow(self, rhs: Oklch) -> Self::Output {
        Oklch::new(self.l.powf(rhs.l), self.c.powf(rhs.c), self.h.powf(rhs.h))
    }
}

impl Oklch {
    #[inline]
    pub fn sqrt(&self) -> Oklch {
        Oklch::new(
            if self.l < 0. { 0. } else { self.l.sqrt() },
            if self.c < 0. { 0. } else { self.c.sqrt() },
            if self.h < 0. { 0. } else { self.h.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Oklch {
        Oklch::new(self.l.cbrt(), self.c.cbrt(), self.h.cbrt())
    }
}
