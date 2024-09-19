/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::Rgb;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, PartialOrd, PartialEq, Copy, Clone)]
/// Represents color as sigmoid function: `y = 1 / (1 + exp(-x))`
/// and it's inverse
/// `x = ln(y / (1 - y))`
pub struct Sigmoidal {
    pub sr: f32,
    pub sg: f32,
    pub sb: f32,
}

#[inline]
fn to_sigmoidal(x: f32) -> f32 {
    let den = 1f32 + (-x).exp();
    if den == 0f32 {
        return 0f32;
    }
    1f32 / den
}

#[inline]
fn inverse_sigmoidal(x: f32) -> f32 {
    let den = 1f32 - x;
    if den == 0f32 {
        return 0f32;
    }
    let k = x / den;
    if k <= 0f32 {
        return 0f32;
    }
    k.ln()
}

impl Sigmoidal {
    #[inline]
    pub fn new(sr: f32, sg: f32, sb: f32) -> Self {
        Sigmoidal { sr, sg, sb }
    }

    #[inline]
    pub fn from_rgb(rgb: Rgb<u8>) -> Self {
        let normalized = rgb.to_rgb_f32();
        Sigmoidal::new(
            to_sigmoidal(normalized.r),
            to_sigmoidal(normalized.g),
            to_sigmoidal(normalized.b),
        )
    }

    #[inline]
    pub fn to_rgb(&self) -> Rgb<u8> {
        let rgb_normalized = Rgb::new(
            inverse_sigmoidal(self.sr),
            inverse_sigmoidal(self.sg),
            inverse_sigmoidal(self.sb),
        );
        rgb_normalized.into()
    }
}

impl From<Rgb<u8>> for Sigmoidal {
    #[inline]
    fn from(value: Rgb<u8>) -> Self {
        Sigmoidal::from_rgb(value)
    }
}

impl From<Rgb<f32>> for Sigmoidal {
    #[inline]
    fn from(value: Rgb<f32>) -> Self {
        Sigmoidal::new(
            to_sigmoidal(value.r),
            to_sigmoidal(value.g),
            to_sigmoidal(value.b),
        )
    }
}

impl Index<usize> for Sigmoidal {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.sr,
            1 => &self.sg,
            2 => &self.sb,
            _ => panic!("Index out of bounds for Sigmoidal"),
        }
    }
}

impl IndexMut<usize> for Sigmoidal {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.sr,
            1 => &mut self.sg,
            2 => &mut self.sb,
            _ => panic!("Index out of bounds for Sigmoidal"),
        }
    }
}

impl Add<f32> for Sigmoidal {
    type Output = Sigmoidal;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Sigmoidal::new(self.sr + rhs, self.sg + rhs, self.sb + rhs)
    }
}

impl Sub<f32> for Sigmoidal {
    type Output = Sigmoidal;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Sigmoidal::new(self.sr - rhs, self.sg - rhs, self.sb - rhs)
    }
}

impl Mul<f32> for Sigmoidal {
    type Output = Sigmoidal;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Sigmoidal::new(self.sr * rhs, self.sg * rhs, self.sb * rhs)
    }
}

impl Div<f32> for Sigmoidal {
    type Output = Sigmoidal;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Sigmoidal::new(self.sr / rhs, self.sg / rhs, self.sb / rhs)
    }
}

impl Add<Sigmoidal> for Sigmoidal {
    type Output = Sigmoidal;

    #[inline]
    fn add(self, rhs: Sigmoidal) -> Self::Output {
        Sigmoidal::new(self.sr + rhs.sr, self.sg + rhs.sg, self.sb + rhs.sb)
    }
}

impl Sub<Sigmoidal> for Sigmoidal {
    type Output = Sigmoidal;

    #[inline]
    fn sub(self, rhs: Sigmoidal) -> Self::Output {
        Sigmoidal::new(self.sr - rhs.sr, self.sg - rhs.sg, self.sb - rhs.sb)
    }
}

impl Mul<Sigmoidal> for Sigmoidal {
    type Output = Sigmoidal;

    #[inline]
    fn mul(self, rhs: Sigmoidal) -> Self::Output {
        Sigmoidal::new(self.sr * rhs.sr, self.sg * rhs.sg, self.sb * rhs.sb)
    }
}

impl Div<Sigmoidal> for Sigmoidal {
    type Output = Sigmoidal;

    #[inline]
    fn div(self, rhs: Sigmoidal) -> Self::Output {
        Sigmoidal::new(self.sr / rhs.sr, self.sg / rhs.sg, self.sb / rhs.sb)
    }
}

impl AddAssign<Sigmoidal> for Sigmoidal {
    #[inline]
    fn add_assign(&mut self, rhs: Sigmoidal) {
        self.sr += rhs.sr;
        self.sg += rhs.sg;
        self.sb += rhs.sb;
    }
}

impl SubAssign<Sigmoidal> for Sigmoidal {
    #[inline]
    fn sub_assign(&mut self, rhs: Sigmoidal) {
        self.sr -= rhs.sr;
        self.sg -= rhs.sg;
        self.sb -= rhs.sb;
    }
}

impl MulAssign<Sigmoidal> for Sigmoidal {
    #[inline]
    fn mul_assign(&mut self, rhs: Sigmoidal) {
        self.sr *= rhs.sr;
        self.sg *= rhs.sg;
        self.sb *= rhs.sb;
    }
}

impl DivAssign<Sigmoidal> for Sigmoidal {
    #[inline]
    fn div_assign(&mut self, rhs: Sigmoidal) {
        self.sr /= rhs.sr;
        self.sg /= rhs.sg;
        self.sb /= rhs.sb;
    }
}

impl AddAssign<f32> for Sigmoidal {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.sr += rhs;
        self.sg += rhs;
        self.sb += rhs;
    }
}

impl SubAssign<f32> for Sigmoidal {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.sr -= rhs;
        self.sg -= rhs;
        self.sb -= rhs;
    }
}

impl MulAssign<f32> for Sigmoidal {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.sr *= rhs;
        self.sg *= rhs;
        self.sb *= rhs;
    }
}

impl DivAssign<f32> for Sigmoidal {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.sr /= rhs;
        self.sg /= rhs;
        self.sb /= rhs;
    }
}
