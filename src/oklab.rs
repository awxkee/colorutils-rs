/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#![allow(clippy::excessive_precision)]
use crate::utils::mlaf;
use crate::{EuclideanDistance, Rgb, TaxicabDistance, TransferFunction};
use num_traits::Pow;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
/// Struct that represent *Oklab* colorspace
pub struct Oklab {
    /// All values in Oklab intended to be normalized [0;1]
    pub l: f32,
    /// A value range [-0.5; 0.5]
    pub a: f32,
    /// B value range [-0.5; 0.5]
    pub b: f32,
}

impl Oklab {
    #[inline]
    pub fn new(l: f32, a: f32, b: f32) -> Oklab {
        Oklab { l, a, b }
    }

    #[inline]
    /// Converts from sRGB [Rgb] to [Oklab] using sRGB transfer function
    pub fn from_srgb(rgb: Rgb<u8>) -> Oklab {
        let rgb_float = rgb.to_rgb_f32();
        let linearized = rgb_float.linearize(TransferFunction::Srgb);
        Self::linear_rgb_to_oklab(linearized)
    }

    #[inline]
    /// Converts from Rgb to [Oklab] using provided [TransferFunction]
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> Oklab {
        let rgb_float = rgb.to_rgb_f32();
        let linearized = rgb_float.linearize(transfer_function);
        Self::linear_rgb_to_oklab(linearized)
    }

    #[inline]
    /// Convert Linear Rgb to [Oklab]
    pub fn from_linear_rgb(rgb: Rgb<f32>) -> Oklab {
        Self::linear_rgb_to_oklab(rgb)
    }

    #[inline]
    /// Converts [Oklab] to [Rgb] using sRGB transfer function
    pub fn to_srgb(&self) -> Rgb<u8> {
        let linear_rgb = self.to_linear_rgb();
        let transferred = linear_rgb.gamma(TransferFunction::Srgb);
        transferred.to_u8()
    }

    #[inline]
    /// Converts [Oklab] to [Rgb] using provided [TransferFunction]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let linear_rgb = self.to_linear_rgb();
        let transferred = linear_rgb.gamma(transfer_function);
        transferred.to_u8()
    }

    #[inline]
    /// Converts [Oklab] to linear [Rgb] using sRGB transfer function
    pub fn to_srgb_f32(&self) -> Rgb<f32> {
        let linear_rgb = self.to_linear_rgb();
        linear_rgb.gamma(TransferFunction::Srgb)
    }

    #[inline]
    /// Converts [Oklab] to [Rgb] using provided [TransferFunction]
    pub fn to_rgb_f32(&self, transfer_function: TransferFunction) -> Rgb<f32> {
        let linear_rgb = self.to_linear_rgb();
        linear_rgb.gamma(transfer_function)
    }

    #[inline]
    fn linear_rgb_to_oklab(rgb: Rgb<f32>) -> Oklab {
        let l = mlaf(
            mlaf(0.4122214708f32 * rgb.r, 0.5363325363f32, rgb.g),
            0.0514459929f32,
            rgb.b,
        );
        let m = mlaf(
            mlaf(0.2119034982f32 * rgb.r, 0.6806995451f32, rgb.g),
            0.1073969566f32,
            rgb.b,
        );
        let s = mlaf(
            mlaf(0.0883024619f32 * rgb.r, 0.2817188376f32, rgb.g),
            0.6299787005f32,
            rgb.b,
        );

        let l_ = l.cbrt();
        let m_ = m.cbrt();
        let s_ = s.cbrt();

        Oklab {
            l: mlaf(
                mlaf(0.2104542553f32 * l_, 0.7936177850f32, m_),
                -0.0040720468f32,
                s_,
            ),
            a: mlaf(
                mlaf(1.9779984951f32 * l_, -2.4285922050f32, m_),
                0.4505937099f32,
                s_,
            ),
            b: mlaf(
                mlaf(0.0259040371f32 * l_, 0.7827717662f32, m_),
                -0.8086757660f32,
                s_,
            ),
        }
    }

    #[inline]
    /// Converts to linear RGB
    pub fn to_linear_rgb(&self) -> Rgb<f32> {
        let l_ = mlaf(
            mlaf(self.l, 0.3963377774f32, self.a),
            0.2158037573f32,
            self.b,
        );
        let m_ = mlaf(
            mlaf(self.l, -0.1055613458f32, self.a),
            -0.0638541728f32,
            self.b,
        );
        let s_ = mlaf(
            mlaf(self.l, -0.0894841775f32, self.a),
            -1.2914855480f32,
            self.b,
        );

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        Rgb::new(
            mlaf(
                mlaf(4.0767416621f32 * l, -3.3077115913f32, m),
                0.2309699292f32,
                s,
            ),
            mlaf(
                mlaf(-1.2684380046f32 * l, 2.6097574011f32, m),
                -0.3413193965f32,
                s,
            ),
            mlaf(
                mlaf(-0.0041960863f32 * l, -0.7034186147f32, m),
                1.7076147010f32,
                s,
            ),
        )
    }

    #[inline]
    pub fn hybrid_distance(&self, other: Self) -> f32 {
        let lax = self.l - other.l;
        let dax = self.a - other.a;
        let bax = self.b - other.b;
        (dax * dax + bax * bax).sqrt() + lax.abs()
    }

    pub const fn l_range() -> (f32, f32) {
        (0., 1.)
    }

    pub const fn a_range() -> (f32, f32) {
        (-0.5, 0.5)
    }

    pub const fn b_range() -> (f32, f32) {
        (-0.5, 0.5)
    }
}

impl EuclideanDistance for Oklab {
    fn euclidean_distance(&self, other: Self) -> f32 {
        let lax = self.l - other.l;
        let dax = self.a - other.a;
        let bax = self.b - other.b;
        (lax * lax + dax * dax + bax * bax).sqrt()
    }
}

impl TaxicabDistance for Oklab {
    fn taxicab_distance(&self, other: Self) -> f32 {
        let lax = self.l - other.l;
        let dax = self.a - other.a;
        let bax = self.b - other.b;
        lax.abs() + dax.abs() + bax.abs()
    }
}

impl Add<Oklab> for Oklab {
    type Output = Oklab;

    #[inline]
    fn add(self, rhs: Self) -> Oklab {
        Oklab::new(self.l + rhs.l, self.a + rhs.a, self.b + rhs.b)
    }
}

impl Add<f32> for Oklab {
    type Output = Oklab;

    #[inline]
    fn add(self, rhs: f32) -> Oklab {
        Oklab::new(self.l + rhs, self.a + rhs, self.b + rhs)
    }
}

impl AddAssign<Oklab> for Oklab {
    #[inline]
    fn add_assign(&mut self, rhs: Oklab) {
        self.l += rhs.l;
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl AddAssign<f32> for Oklab {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.l += rhs;
        self.a += rhs;
        self.b += rhs;
    }
}

impl Mul<f32> for Oklab {
    type Output = Oklab;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Oklab::new(self.l * rhs, self.a * rhs, self.b * rhs)
    }
}

impl Mul<Oklab> for Oklab {
    type Output = Oklab;

    #[inline]
    fn mul(self, rhs: Oklab) -> Self::Output {
        Oklab::new(self.l * rhs.l, self.a * rhs.a, self.b * rhs.b)
    }
}

impl MulAssign<f32> for Oklab {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.l *= rhs;
        self.a *= rhs;
        self.b *= rhs;
    }
}

impl MulAssign<Oklab> for Oklab {
    #[inline]
    fn mul_assign(&mut self, rhs: Oklab) {
        self.l *= rhs.l;
        self.a *= rhs.a;
        self.b *= rhs.b;
    }
}

impl Sub<f32> for Oklab {
    type Output = Oklab;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Oklab::new(self.l - rhs, self.a - rhs, self.b - rhs)
    }
}

impl Sub<Oklab> for Oklab {
    type Output = Oklab;

    #[inline]
    fn sub(self, rhs: Oklab) -> Self::Output {
        Oklab::new(self.l - rhs.l, self.a - rhs.a, self.b - rhs.b)
    }
}

impl SubAssign<f32> for Oklab {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.l -= rhs;
        self.a -= rhs;
        self.b -= rhs;
    }
}

impl SubAssign<Oklab> for Oklab {
    #[inline]
    fn sub_assign(&mut self, rhs: Oklab) {
        self.l -= rhs.l;
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl Div<f32> for Oklab {
    type Output = Oklab;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Oklab::new(self.l / rhs, self.a / rhs, self.b / rhs)
    }
}

impl Div<Oklab> for Oklab {
    type Output = Oklab;

    #[inline]
    fn div(self, rhs: Oklab) -> Self::Output {
        Oklab::new(self.l / rhs.l, self.a / rhs.a, self.b / rhs.b)
    }
}

impl DivAssign<f32> for Oklab {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.l /= rhs;
        self.a /= rhs;
        self.b /= rhs;
    }
}

impl DivAssign<Oklab> for Oklab {
    #[inline]
    fn div_assign(&mut self, rhs: Oklab) {
        self.l /= rhs.l;
        self.a /= rhs.a;
        self.b /= rhs.b;
    }
}

impl Neg for Oklab {
    type Output = Oklab;

    #[inline]
    fn neg(self) -> Self::Output {
        Oklab::new(-self.l, -self.a, -self.b)
    }
}

impl Pow<f32> for Oklab {
    type Output = Oklab;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Oklab::new(self.l.powf(rhs), self.a.powf(rhs), self.b.powf(rhs))
    }
}

impl Pow<Oklab> for Oklab {
    type Output = Oklab;

    #[inline]
    fn pow(self, rhs: Oklab) -> Self::Output {
        Oklab::new(self.l.powf(rhs.l), self.a.powf(rhs.a), self.b.powf(rhs.b))
    }
}

impl Oklab {
    #[inline]
    pub fn sqrt(&self) -> Oklab {
        Oklab::new(
            if self.l < 0. { 0. } else { self.l.sqrt() },
            if self.a < 0. { 0. } else { self.a.sqrt() },
            if self.b < 0. { 0. } else { self.b.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Oklab {
        Oklab::new(self.l.cbrt(), self.a.cbrt(), self.b.cbrt())
    }
}
