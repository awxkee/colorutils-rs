/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::utils::mlaf;
use crate::{EuclideanDistance, Rgb, TaxicabDistance, TransferFunction};
use num_traits::Pow;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// XYB is a color space that was designed for use with the JPEG XL Image Coding System.
///
/// It is an LMS-based color model inspired by the human visual system, facilitating perceptually uniform quantization.
/// It uses a gamma of 3 for computationally efficient decoding.
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub struct Xyb {
    pub x: f32,
    pub y: f32,
    pub b: f32,
}

impl Xyb {
    #[inline]
    pub fn new(x: f32, y: f32, b: f32) -> Xyb {
        Xyb { x, y, b }
    }

    #[inline]
    /// Converts [Rgb] to [Xyb] using provided [TransferFunction]
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> Xyb {
        let linear_rgb = rgb.to_linear(transfer_function);
        Self::from_linear_rgb(linear_rgb)
    }

    #[inline]
    /// Converts linear [Rgb] to [Xyb]
    pub fn from_linear_rgb(rgb: Rgb<f32>) -> Xyb {
        const BIAS_CBRT: f32 = 0.155954200549248620f32;
        const BIAS: f32 = 0.00379307325527544933;
        let lgamma = mlaf(
            0.3f32,
            rgb.r,
            mlaf(0.622f32, rgb.g, mlaf(0.078f32, rgb.b, BIAS)),
        )
        .cbrt()
            - BIAS_CBRT;
        let mgamma = mlaf(
            0.23f32,
            rgb.r,
            mlaf(0.692f32, rgb.g, mlaf(0.078f32, rgb.b, BIAS)),
        )
        .cbrt()
            - BIAS_CBRT;
        let sgamma = mlaf(
            0.24342268924547819f32,
            rgb.r,
            mlaf(
                0.20476744424496821f32,
                rgb.g,
                mlaf(0.55180986650955360f32, rgb.b, BIAS),
            ),
        )
        .cbrt()
            - BIAS_CBRT;
        let x = (lgamma - mgamma) * 0.5f32;
        let y = (lgamma + mgamma) * 0.5f32;
        let b = sgamma - mgamma;
        Xyb::new(x, y, b)
    }

    #[inline]
    /// Converts [Xyb] to linear [Rgb]
    pub fn to_linear_rgb(&self) -> Rgb<f32> {
        const BIAS_CBRT: f32 = 0.155954200549248620f32;
        const BIAS: f32 = 0.00379307325527544933;
        let x_lms = (self.x + self.y) + BIAS_CBRT;
        let y_lms = (-self.x + self.y) + BIAS_CBRT;
        let b_lms = (-self.x + self.y + self.b) + BIAS_CBRT;
        let x_c_lms = (x_lms * x_lms * x_lms) - BIAS;
        let y_c_lms = (y_lms * y_lms * y_lms) - BIAS;
        let b_c_lms = (b_lms * b_lms * b_lms) - BIAS;
        let r = mlaf(
            11.031566901960783,
            x_c_lms,
            mlaf(-9.866943921568629, y_c_lms, -0.16462299647058826 * b_c_lms),
        );
        let g = mlaf(
            -3.254147380392157,
            x_c_lms,
            mlaf(4.418770392156863, y_c_lms, -0.16462299647058826 * b_c_lms),
        );
        let b = mlaf(
            -3.6588512862745097,
            x_c_lms,
            mlaf(2.7129230470588235, y_c_lms, 1.9459282392156863 * b_c_lms),
        );
        Rgb::new(r, g, b)
    }

    #[inline]
    /// Converts [Xyb] to [Rgb] using provided [TransferFunction]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let linear_rgb = self.to_linear_rgb();
        linear_rgb
            .apply(transfer_function.get_gamma_function())
            .to_u8()
    }
}

impl Add<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x + rhs, self.y + rhs, self.b + rhs)
    }
}

impl Add<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn add(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x + rhs.x, self.y + rhs.y, self.b + rhs.b)
    }
}

impl Sub<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x - rhs, self.y - rhs, self.b - rhs)
    }
}

impl Sub<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn sub(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x - rhs.x, self.y - rhs.y, self.b - rhs.b)
    }
}

impl Mul<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x * rhs, self.y * rhs, self.b * rhs)
    }
}

impl Mul<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn mul(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x * rhs.x, self.y * rhs.y, self.b * rhs.b)
    }
}

impl Div<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x / rhs, self.y / rhs, self.b / rhs)
    }
}

impl Div<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn div(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x / rhs.x, self.y / rhs.y, self.b / rhs.b)
    }
}

impl Neg for Xyb {
    type Output = Xyb;

    #[inline]
    fn neg(self) -> Self::Output {
        Xyb::new(-self.x, -self.y, -self.b)
    }
}

impl Pow<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x.powf(rhs), self.y.powf(rhs), self.b.powf(rhs))
    }
}

impl Pow<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn pow(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x.powf(rhs.x), self.y.powf(rhs.y), self.b.powf(rhs.b))
    }
}

impl MulAssign<f32> for Xyb {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.b *= rhs;
    }
}

impl MulAssign<Xyb> for Xyb {
    #[inline]
    fn mul_assign(&mut self, rhs: Xyb) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.b *= rhs.b;
    }
}

impl AddAssign<f32> for Xyb {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.b += rhs;
    }
}

impl AddAssign<Xyb> for Xyb {
    #[inline]
    fn add_assign(&mut self, rhs: Xyb) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.b += rhs.b;
    }
}

impl SubAssign<f32> for Xyb {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.b -= rhs;
    }
}

impl SubAssign<Xyb> for Xyb {
    #[inline]
    fn sub_assign(&mut self, rhs: Xyb) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.b -= rhs.b;
    }
}

impl DivAssign<f32> for Xyb {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.b /= rhs;
    }
}

impl DivAssign<Xyb> for Xyb {
    #[inline]
    fn div_assign(&mut self, rhs: Xyb) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.b /= rhs.b;
    }
}

impl Xyb {
    #[inline]
    pub fn sqrt(&self) -> Xyb {
        Xyb::new(
            if self.x < 0. { 0. } else { self.x.sqrt() },
            if self.y < 0. { 0. } else { self.y.sqrt() },
            if self.b < 0. { 0. } else { self.b.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Xyb {
        Xyb::new(self.x.cbrt(), self.y.cbrt(), self.b.cbrt())
    }
}

impl EuclideanDistance for Xyb {
    fn euclidean_distance(&self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let db = self.b - other.b;
        (dx * dx + dy * dy + db * db).sqrt()
    }
}

impl TaxicabDistance for Xyb {
    fn taxicab_distance(&self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let db = self.b - other.b;
        dx.abs() + dy.abs() + db.abs()
    }
}
