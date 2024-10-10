/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

//! # Luv
/// Struct representing a color in CIE LUV, a.k.a. L\*u\*v\*, color space
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialOrd)]
pub struct Luv {
    /// The L\* value (achromatic luminance) of the colour in 0–100 range.
    pub l: f32,
    /// The u\* value of the colour.
    ///
    /// Together with v\* value, it defines chromaticity of the colour.  The u\*
    /// coordinate represents colour’s position on red-green axis with negative
    /// values indicating more red and positive more green colour.  Typical
    /// values are in -134–220 range (but exact range for ‘valid’ colours
    /// depends on luminance and v\* value).
    pub u: f32,
    /// The u\* value of the colour.
    ///
    /// Together with u\* value, it defines chromaticity of the colour.  The v\*
    /// coordinate represents colour’s position on blue-yellow axis with
    /// negative values indicating more blue and positive more yellow colour.
    /// Typical values are in -140–122 range (but exact range for ‘valid’
    /// colours depends on luminance and u\* value).
    pub v: f32,
}

/// Representing a color in cylindrical CIE LCh(uv) color space
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialOrd)]
pub struct LCh {
    /// The L\* value (achromatic luminance) of the colour in 0–100 range.
    ///
    /// This is the same value as in the [`Luv`] object.
    pub l: f32,
    /// The C\*_uv value (chroma) of the colour.
    ///
    /// Together with h_uv, it defines chromaticity of the colour.  The typical
    /// values of the coordinate go from zero up to around 150 (but exact range
    /// for ‘valid’ colours depends on luminance and hue).  Zero represents
    /// shade of grey.
    pub c: f32,
    /// The h_uv value (hue) of the colour measured in radians.
    ///
    /// Together with C\*_uv, it defines chromaticity of the colour.  The value
    /// represents an angle thus it wraps around τ.  Typically, the value will
    /// be in the -π–π range.  The value is undefined if C\*_uv is zero.
    pub h: f32,
}

const D65_XYZ: [f32; 3] = [95.047f32, 100.0f32, 108.883f32];

use crate::rgb::Rgb;
use crate::rgba::Rgba;
use crate::xyz::Xyz;
use crate::{EuclideanDistance, TaxicabDistance};
use num_traits::Pow;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub(crate) const LUV_WHITE_U_PRIME: f32 =
    4.0f32 * D65_XYZ[1] / (D65_XYZ[0] + 15.0 * D65_XYZ[1] + 3.0 * D65_XYZ[2]);
pub(crate) const LUV_WHITE_V_PRIME: f32 =
    9.0f32 * D65_XYZ[1] / (D65_XYZ[0] + 15.0 * D65_XYZ[1] + 3.0 * D65_XYZ[2]);

pub(crate) const LUV_CUTOFF_FORWARD_Y: f32 = (6f32 / 29f32) * (6f32 / 29f32) * (6f32 / 29f32);
pub(crate) const LUV_MULTIPLIER_FORWARD_Y: f32 = (29f32 / 3f32) * (29f32 / 3f32) * (29f32 / 3f32);
pub(crate) const LUV_MULTIPLIER_INVERSE_Y: f32 = (3f32 / 29f32) * (3f32 / 29f32) * (3f32 / 29f32);
impl Luv {
    /// Converts RGB to CIE Luv
    #[inline]
    #[allow(clippy::manual_clamp)]
    pub fn from_rgb(rgb: Rgb<u8>) -> Self {
        let xyz = Xyz::from_srgb(rgb);
        Self::from_xyz(xyz)
    }

    /// Converts CIE XYZ to CIE Luv
    #[inline]
    #[allow(clippy::manual_clamp)]
    pub fn from_xyz(xyz: Xyz) -> Self {
        let [x, y, z] = [xyz.x, xyz.y, xyz.z];
        let den = x + 15.0 * y + 3.0 * z;

        let l = (if y < LUV_CUTOFF_FORWARD_Y {
            LUV_MULTIPLIER_FORWARD_Y * y
        } else {
            116f32 * y.cbrt() - 16f32
        })
        .min(100f32)
        .max(0f32);
        let (u, v);
        if den != 0f32 {
            let u_prime = 4f32 * x / den;
            let v_prime = 9f32 * y / den;
            u = 13f32 * l * (u_prime - LUV_WHITE_U_PRIME);
            v = 13f32 * l * (v_prime - LUV_WHITE_V_PRIME);
        } else {
            u = 0f32;
            v = 0f32;
        }

        Luv { l, u, v }
    }

    #[inline]
    pub fn from_rgba(rgba: Rgba<u8>) -> Self {
        Luv::from_rgb(rgba.to_rgb())
    }

    #[inline]
    pub fn to_xyz(&self) -> Xyz {
        if self.l <= 0f32 {
            return Xyz::new(0f32, 0f32, 0f32);
        }
        let l13 = 1f32 / (13f32 * self.l);
        let u = self.u * l13 + LUV_WHITE_U_PRIME;
        let v = self.v * l13 + LUV_WHITE_V_PRIME;
        let y = if self.l > 8f32 {
            ((self.l + 16f32) / 116f32).powi(3)
        } else {
            self.l * LUV_MULTIPLIER_INVERSE_Y
        };
        let (x, z);
        if v != 0f32 {
            let den = 1f32 / (4f32 * v);
            x = y * 9f32 * u * den;
            z = y * (12.0f32 - 3.0f32 * u - 20f32 * v) * den;
        } else {
            x = 0f32;
            z = 0f32;
        }

        Xyz::new(x, y, z)
    }

    /// Converts CIE [Luv] into linear [Rgb]
    #[inline]
    pub fn to_linear_rgb(&self, matrix: &[[f32; 3]; 3]) -> Rgb<f32> {
        let xyz = self.to_xyz();
        xyz.to_linear_rgb(matrix)
    }

    #[inline]
    pub fn to_rgb(&self) -> Rgb<u8> {
        let xyz = self.to_xyz();
        xyz.to_srgb()
    }

    pub fn new(l: f32, u: f32, v: f32) -> Luv {
        Luv { l, u, v }
    }

    pub const fn l_range() -> (f32, f32) {
        (0., 100.)
    }

    pub const fn u_range() -> (f32, f32) {
        (-100., 100.)
    }

    pub const fn v_range() -> (f32, f32) {
        (-100., 100.)
    }
}

impl LCh {
    pub fn from_rgb(rgb: Rgb<u8>) -> Self {
        LCh::from_luv(Luv::from_rgb(rgb))
    }

    pub fn from_rgba(rgba: Rgba<u8>) -> Self {
        LCh::from_luv(Luv::from_rgba(rgba))
    }

    pub fn new(l: f32, c: f32, h: f32) -> Self {
        LCh { l, c, h }
    }

    pub fn from_luv(luv: Luv) -> Self {
        LCh {
            l: luv.l,
            c: luv.u.hypot(luv.v),
            h: luv.v.atan2(luv.u),
        }
    }

    #[inline]
    pub fn to_rgb(&self) -> Rgb<u8> {
        self.to_luv().to_rgb()
    }

    #[inline]
    pub fn to_xyz(&self) -> Xyz {
        self.to_luv().to_xyz()
    }

    #[inline]
    pub fn to_linear_rgb(&self, matrix: &[[f32; 3]; 3]) -> Rgb<f32> {
        let xyz = self.to_xyz();
        xyz.to_linear_rgb(matrix)
    }

    #[inline]
    pub fn to_luv(&self) -> Luv {
        Luv {
            l: self.l,
            u: self.c * self.h.cos(),
            v: self.c * self.h.sin(),
        }
    }
}

impl PartialEq<Luv> for Luv {
    /// Compares two colours ignoring chromaticity if L\* is zero.
    fn eq(&self, other: &Self) -> bool {
        if self.l != other.l {
            false
        } else if self.l == 0.0 {
            true
        } else {
            self.u == other.u && self.v == other.v
        }
    }
}

impl PartialEq<LCh> for LCh {
    /// Compares two colours ignoring chromaticity if L\* is zero and hue if C\*
    /// is zero.  Hues which are τ apart are compared equal.
    fn eq(&self, other: &Self) -> bool {
        if self.l != other.l {
            false
        } else if self.l == 0.0 {
            true
        } else if self.c != other.c {
            false
        } else if self.c == 0.0 {
            true
        } else {
            use std::f32::consts::TAU;
            self.h.rem_euclid(TAU) == other.h.rem_euclid(TAU)
        }
    }
}

impl EuclideanDistance for Luv {
    #[inline]
    fn euclidean_distance(&self, other: Luv) -> f32 {
        let dl = self.l - other.l;
        let du = self.u - other.u;
        let dv = self.v - other.v;
        (dl * dl + du * du + dv * dv).sqrt()
    }
}

impl EuclideanDistance for LCh {
    #[inline]
    fn euclidean_distance(&self, other: LCh) -> f32 {
        let dl = self.l - other.l;
        let dc = self.c - other.c;
        let dh = self.h - other.h;
        (dl * dl + dc * dc + dh * dh).sqrt()
    }
}

impl TaxicabDistance for Luv {
    #[inline]
    fn taxicab_distance(&self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let du = self.u - other.u;
        let dv = self.v - other.v;
        dl.abs() + du.abs() + dv.abs()
    }
}

impl TaxicabDistance for LCh {
    #[inline]
    fn taxicab_distance(&self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let dc = self.c - other.c;
        let dh = self.h - other.h;
        dl.abs() + dc.abs() + dh.abs()
    }
}

impl Add<Luv> for Luv {
    type Output = Luv;

    #[inline]
    fn add(self, rhs: Luv) -> Luv {
        Luv::new(self.l + rhs.l, self.u + rhs.u, self.v + rhs.v)
    }
}

impl Add<LCh> for LCh {
    type Output = LCh;

    #[inline]
    fn add(self, rhs: LCh) -> LCh {
        LCh::new(self.l + rhs.l, self.c + rhs.c, self.h + rhs.h)
    }
}

impl Sub<Luv> for Luv {
    type Output = Luv;

    #[inline]
    fn sub(self, rhs: Luv) -> Luv {
        Luv::new(self.l - rhs.l, self.u - rhs.u, self.v - rhs.v)
    }
}

impl Sub<LCh> for LCh {
    type Output = LCh;

    #[inline]
    fn sub(self, rhs: LCh) -> LCh {
        LCh::new(self.l - rhs.l, self.c - rhs.c, self.h - rhs.h)
    }
}

impl Mul<Luv> for Luv {
    type Output = Luv;

    #[inline]
    fn mul(self, rhs: Luv) -> Luv {
        Luv::new(self.l * rhs.l, self.u * rhs.u, self.v * rhs.v)
    }
}

impl Mul<LCh> for LCh {
    type Output = LCh;

    #[inline]
    fn mul(self, rhs: LCh) -> LCh {
        LCh::new(self.l * rhs.l, self.c * rhs.c, self.h * rhs.h)
    }
}

impl Div<Luv> for Luv {
    type Output = Luv;

    #[inline]
    fn div(self, rhs: Luv) -> Luv {
        Luv::new(self.l / rhs.l, self.u / rhs.u, self.v / rhs.v)
    }
}

impl Div<LCh> for LCh {
    type Output = LCh;

    #[inline]
    fn div(self, rhs: LCh) -> LCh {
        LCh::new(self.l / rhs.l, self.c / rhs.c, self.h / rhs.h)
    }
}

impl Add<f32> for Luv {
    type Output = Luv;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Luv::new(self.l + rhs, self.u + rhs, self.v + rhs)
    }
}

impl Add<f32> for LCh {
    type Output = LCh;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        LCh::new(self.l + rhs, self.c + rhs, self.h + rhs)
    }
}

impl Sub<f32> for Luv {
    type Output = Luv;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Luv::new(self.l - rhs, self.u - rhs, self.v - rhs)
    }
}

impl Sub<f32> for LCh {
    type Output = LCh;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        LCh::new(self.l - rhs, self.c - rhs, self.h - rhs)
    }
}

impl Mul<f32> for Luv {
    type Output = Luv;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Luv::new(self.l * rhs, self.u * rhs, self.v * rhs)
    }
}

impl Mul<f32> for LCh {
    type Output = LCh;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        LCh::new(self.l * rhs, self.c * rhs, self.h * rhs)
    }
}

impl Div<f32> for Luv {
    type Output = Luv;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Luv::new(self.l / rhs, self.u / rhs, self.v / rhs)
    }
}

impl Div<f32> for LCh {
    type Output = LCh;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        LCh::new(self.l / rhs, self.c / rhs, self.h / rhs)
    }
}

impl AddAssign<Luv> for Luv {
    #[inline]
    fn add_assign(&mut self, rhs: Luv) {
        self.l += rhs.l;
        self.u += rhs.u;
        self.v += rhs.v;
    }
}

impl AddAssign<LCh> for LCh {
    #[inline]
    fn add_assign(&mut self, rhs: LCh) {
        self.l += rhs.l;
        self.c += rhs.c;
        self.h += rhs.h;
    }
}

impl SubAssign<Luv> for Luv {
    #[inline]
    fn sub_assign(&mut self, rhs: Luv) {
        self.l -= rhs.l;
        self.u -= rhs.u;
        self.v -= rhs.v;
    }
}

impl SubAssign<LCh> for LCh {
    #[inline]
    fn sub_assign(&mut self, rhs: LCh) {
        self.l -= rhs.l;
        self.c -= rhs.c;
        self.h -= rhs.h;
    }
}

impl MulAssign<Luv> for Luv {
    #[inline]
    fn mul_assign(&mut self, rhs: Luv) {
        self.l *= rhs.l;
        self.u *= rhs.u;
        self.v *= rhs.v;
    }
}

impl MulAssign<LCh> for LCh {
    #[inline]
    fn mul_assign(&mut self, rhs: LCh) {
        self.l *= rhs.l;
        self.c *= rhs.c;
        self.h *= rhs.h;
    }
}

impl DivAssign<Luv> for Luv {
    #[inline]
    fn div_assign(&mut self, rhs: Luv) {
        self.l /= rhs.l;
        self.u /= rhs.u;
        self.v /= rhs.v;
    }
}

impl DivAssign<LCh> for LCh {
    #[inline]
    fn div_assign(&mut self, rhs: LCh) {
        self.l /= rhs.l;
        self.c /= rhs.c;
        self.h /= rhs.h;
    }
}

impl AddAssign<f32> for Luv {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.l += rhs;
        self.u += rhs;
        self.v += rhs;
    }
}

impl AddAssign<f32> for LCh {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.l += rhs;
        self.c += rhs;
        self.h += rhs;
    }
}

impl SubAssign<f32> for Luv {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.l -= rhs;
        self.u -= rhs;
        self.v -= rhs;
    }
}

impl SubAssign<f32> for LCh {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.l -= rhs;
        self.c -= rhs;
        self.h -= rhs;
    }
}

impl MulAssign<f32> for Luv {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.l *= rhs;
        self.u *= rhs;
        self.v *= rhs;
    }
}

impl MulAssign<f32> for LCh {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.l *= rhs;
        self.c *= rhs;
        self.h *= rhs;
    }
}

impl DivAssign<f32> for Luv {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.l /= rhs;
        self.u /= rhs;
        self.v /= rhs;
    }
}

impl DivAssign<f32> for LCh {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.l /= rhs;
        self.c /= rhs;
        self.h /= rhs;
    }
}

impl Neg for LCh {
    type Output = LCh;

    #[inline]
    fn neg(self) -> Self::Output {
        LCh::new(-self.l, -self.c, -self.h)
    }
}

impl Neg for Luv {
    type Output = Luv;

    #[inline]
    fn neg(self) -> Self::Output {
        Luv::new(-self.l, -self.u, -self.v)
    }
}

impl Pow<f32> for Luv {
    type Output = Luv;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Luv::new(self.l.powf(rhs), self.u.powf(rhs), self.v.powf(rhs))
    }
}

impl Pow<f32> for LCh {
    type Output = LCh;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        LCh::new(self.l.powf(rhs), self.c.powf(rhs), self.h.powf(rhs))
    }
}

impl Pow<Luv> for Luv {
    type Output = Luv;

    #[inline]
    fn pow(self, rhs: Luv) -> Self::Output {
        Luv::new(self.l.powf(rhs.l), self.u.pow(rhs.u), self.v.powf(rhs.v))
    }
}

impl Pow<LCh> for LCh {
    type Output = LCh;

    #[inline]
    fn pow(self, rhs: LCh) -> Self::Output {
        LCh::new(self.l.powf(rhs.l), self.c.powf(rhs.c), self.h.powf(rhs.h))
    }
}

impl Luv {
    #[inline]
    pub fn sqrt(&self) -> Luv {
        Luv::new(
            if self.l < 0. { 0. } else { self.l.sqrt() },
            if self.u < 0. { 0. } else { self.u.sqrt() },
            if self.v < 0. { 0. } else { self.v.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Luv {
        Luv::new(self.l.cbrt(), self.u.cbrt(), self.v.cbrt())
    }
}

impl LCh {
    #[inline]
    pub fn sqrt(&self) -> LCh {
        LCh::new(
            if self.l < 0. { 0. } else { self.l.sqrt() },
            if self.c < 0. { 0. } else { self.c.sqrt() },
            if self.h < 0. { 0. } else { self.h.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> LCh {
        LCh::new(self.l.cbrt(), self.c.cbrt(), self.h.cbrt())
    }
}
