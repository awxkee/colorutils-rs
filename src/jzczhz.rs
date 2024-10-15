/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{EuclideanDistance, Jzazbz, Rgb, TaxicabDistance, TransferFunction, Xyz};
use erydanos::{eatan2f, ehypot3f, ehypotf, Cosine, Sine};
use num_traits::Pow;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Represents Jzazbz in polar coordinates as Jzczhz
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct Jzczhz {
    /// Jz(lightness) generally expects to be between [0;1]
    pub jz: f32,
    /// Cz generally expects to be between [-1;1]
    pub cz: f32,
    /// Hz generally expects to be between [-1;1]
    pub hz: f32,
}

impl Jzczhz {
    /// Creates new instance of Jzczhz
    #[inline]
    pub fn new(jz: f32, cz: f32, hz: f32) -> Jzczhz {
        Jzczhz { jz, cz, hz }
    }

    /// Converts Rgb to polar coordinates Jzczhz
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> Jzczhz {
        let jzazbz = rgb.to_jzazbz(transfer_function);
        Jzczhz::from_jzazbz(jzazbz)
    }

    /// Converts Rgb to polar coordinates Jzczhz
    ///
    /// # Arguments
    /// `display_luminance` - display luminance
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn from_rgb_with_luminance(
        rgb: Rgb<u8>,
        display_luminance: f32,
        transfer_function: TransferFunction,
    ) -> Jzczhz {
        let jzazbz = rgb.to_jzazbz_with_luminance(display_luminance, transfer_function);
        Jzczhz::from_jzazbz(jzazbz)
    }

    /// Converts Jzazbz to polar coordinates Jzczhz
    #[inline]
    pub fn from_jzazbz(jzazbz: Jzazbz) -> Jzczhz {
        let cz = ehypotf(jzazbz.az, jzazbz.bz);
        let hz = eatan2f(jzazbz.bz, jzazbz.az);
        Jzczhz::new(jzazbz.jz, cz, hz)
    }

    /// Converts Jzczhz into Jzazbz
    #[inline]
    pub fn to_jzazbz(&self) -> Jzazbz {
        let az = self.cz * self.hz.ecos();
        let bz = self.cz * self.hz.esin();
        Jzazbz::new(self.jz, az, bz)
    }

    /// Converts Jzczhz into Jzazbz
    #[inline]
    pub fn to_jzazbz_with_luminance(&self, display_luminance: f32) -> Jzazbz {
        let az = self.cz * self.hz.ecos();
        let bz = self.cz * self.hz.esin();
        Jzazbz::new_with_luminance(self.jz, az, bz, display_luminance)
    }

    /// Converts Jzczhz to Rgb
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let jzazbz = self.to_jzazbz();
        jzazbz.to_rgb(transfer_function)
    }

    /// Converts Jzczhz to Rgb
    ///
    /// # Arguments
    /// `display_luminance` - display luminance
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn to_rgb_with_luminance(
        &self,
        display_luminance: f32,
        transfer_function: TransferFunction,
    ) -> Rgb<u8> {
        let jzazbz = self.to_jzazbz_with_luminance(display_luminance);
        jzazbz.to_rgb(transfer_function)
    }

    /// Converts [Jzczhz] to linear [Rgb]
    ///
    /// # Arguments
    /// `display_luminance` - display luminance
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn to_linear_rgb_with_luminance(&self, display_luminance: f32) -> Rgb<f32> {
        let jzazbz = self.to_jzazbz_with_luminance(display_luminance);
        jzazbz.to_linear_rgb()
    }

    /// Converts Jzczhz to *Xyz*
    #[inline]
    pub fn to_xyz(&self) -> Xyz {
        let jzazbz = self.to_jzazbz();
        jzazbz.to_xyz()
    }

    /// Converts [Xyz] to [Jzczhz]
    #[inline]
    pub fn from_xyz(xyz: Xyz) -> Jzczhz {
        let jzazbz = Jzazbz::from_xyz(xyz);
        Jzczhz::from_jzazbz(jzazbz)
    }

    /// Converts [Xyz] to [Jzczhz]
    #[inline]
    pub fn from_xyz_with_display_luminance(xyz: Xyz, luminance: f32) -> Jzczhz {
        let jzazbz = Jzazbz::from_xyz_with_display_luminance(xyz, luminance);
        Jzczhz::from_jzazbz(jzazbz)
    }

    /// Computes distance for *Jzczhz*
    #[inline]
    pub fn distance(&self, other: Jzczhz) -> f32 {
        let djz = self.jz - other.jz;
        let dcz = self.cz - other.cz;
        let dhz = self.hz - other.hz;
        let dh = 2f32 * (self.cz * other.cz).sqrt() * (dhz * 0.5f32).esin();
        ehypot3f(djz, dcz, dh)
    }
}

impl EuclideanDistance for Jzczhz {
    #[inline]
    fn euclidean_distance(&self, other: Self) -> f32 {
        let djz = self.jz - other.jz;
        let dhz = self.hz - other.hz;
        let dcz = self.cz - other.cz;
        (djz * djz + dhz * dhz + dcz * dcz).sqrt()
    }
}

impl TaxicabDistance for Jzczhz {
    #[inline]
    fn taxicab_distance(&self, other: Self) -> f32 {
        let djz = self.jz - other.jz;
        let dhz = self.hz - other.hz;
        let dcz = self.cz - other.cz;
        djz.abs() + dhz.abs() + dcz.abs()
    }
}

impl Index<usize> for Jzczhz {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.jz,
            1 => &self.cz,
            2 => &self.hz,
            _ => panic!("Index out of bounds for Jzczhz"),
        }
    }
}

impl IndexMut<usize> for Jzczhz {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.jz,
            1 => &mut self.cz,
            2 => &mut self.hz,
            _ => panic!("Index out of bounds for Jzczhz"),
        }
    }
}

impl Add<f32> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Jzczhz::new(self.jz + rhs, self.cz + rhs, self.hz + rhs)
    }
}

impl Sub<f32> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Jzczhz::new(self.jz - rhs, self.cz - rhs, self.hz - rhs)
    }
}

impl Mul<f32> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Jzczhz::new(self.jz * rhs, self.cz * rhs, self.hz * rhs)
    }
}

impl Div<f32> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Jzczhz::new(self.jz / rhs, self.cz / rhs, self.hz / rhs)
    }
}

impl Add<Jzczhz> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn add(self, rhs: Jzczhz) -> Self::Output {
        Jzczhz::new(self.jz + rhs.jz, self.cz + rhs.cz, self.hz + rhs.hz)
    }
}

impl Sub<Jzczhz> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn sub(self, rhs: Jzczhz) -> Self::Output {
        Jzczhz::new(self.jz - rhs.jz, self.cz - rhs.cz, self.hz - rhs.hz)
    }
}

impl Mul<Jzczhz> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn mul(self, rhs: Jzczhz) -> Self::Output {
        Jzczhz::new(self.jz * rhs.jz, self.cz * rhs.cz, self.hz * rhs.hz)
    }
}

impl Div<Jzczhz> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn div(self, rhs: Jzczhz) -> Self::Output {
        Jzczhz::new(self.jz / rhs.jz, self.cz / rhs.cz, self.hz / rhs.hz)
    }
}

impl AddAssign<Jzczhz> for Jzczhz {
    #[inline]
    fn add_assign(&mut self, rhs: Jzczhz) {
        self.jz += rhs.jz;
        self.cz += rhs.cz;
        self.hz += rhs.hz;
    }
}

impl SubAssign<Jzczhz> for Jzczhz {
    #[inline]
    fn sub_assign(&mut self, rhs: Jzczhz) {
        self.jz -= rhs.jz;
        self.cz -= rhs.cz;
        self.hz -= rhs.hz;
    }
}

impl MulAssign<Jzczhz> for Jzczhz {
    #[inline]
    fn mul_assign(&mut self, rhs: Jzczhz) {
        self.jz *= rhs.jz;
        self.cz *= rhs.cz;
        self.hz *= rhs.hz;
    }
}

impl DivAssign<Jzczhz> for Jzczhz {
    #[inline]
    fn div_assign(&mut self, rhs: Jzczhz) {
        self.jz /= rhs.jz;
        self.cz /= rhs.cz;
        self.hz /= rhs.hz;
    }
}

impl AddAssign<f32> for Jzczhz {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.jz += rhs;
        self.cz += rhs;
        self.hz += rhs;
    }
}

impl SubAssign<f32> for Jzczhz {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.jz -= rhs;
        self.cz -= rhs;
        self.hz -= rhs;
    }
}

impl MulAssign<f32> for Jzczhz {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.jz *= rhs;
        self.cz *= rhs;
        self.hz *= rhs;
    }
}

impl DivAssign<f32> for Jzczhz {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.jz /= rhs;
        self.cz /= rhs;
        self.hz /= rhs;
    }
}

impl Jzczhz {
    #[inline]
    pub fn sqrt(&self) -> Jzczhz {
        Jzczhz::new(
            if self.jz < 0. { 0. } else { self.jz.sqrt() },
            if self.cz < 0. { 0. } else { self.cz.sqrt() },
            if self.hz < 0. { 0. } else { self.hz.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Jzczhz {
        Jzczhz::new(self.jz.cbrt(), self.cz.cbrt(), self.hz.cbrt())
    }
}

impl Pow<f32> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Jzczhz::new(self.jz.powf(rhs), self.cz.powf(rhs), self.hz.powf(rhs))
    }
}

impl Pow<Jzczhz> for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn pow(self, rhs: Jzczhz) -> Self::Output {
        Jzczhz::new(
            self.jz.powf(rhs.jz),
            self.cz.powf(self.cz),
            self.hz.powf(self.hz),
        )
    }
}

impl Neg for Jzczhz {
    type Output = Jzczhz;

    #[inline]
    fn neg(self) -> Self::Output {
        Jzczhz::new(-self.jz, -self.cz, -self.hz)
    }
}
