/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{
    EuclideanDistance, Jzczhz, Rgb, TaxicabDistance, TransferFunction, Xyz, SRGB_TO_XYZ_D65,
    XYZ_TO_SRGB_D65,
};
use num_traits::Pow;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[inline]
fn perceptual_quantizer(x: f32) -> f32 {
    if x <= 0. {
        return 0.;
    }
    let xx = f32::powf(x * 1e-4, 0.1593017578125);
    let rs = f32::powf(
        (0.8359375 + 18.8515625 * xx) / (1. + 18.6875 * xx),
        134.034375,
    );
    if rs.is_nan() {
        return 0.;
    }
    rs
}

#[inline]
fn perceptual_quantizer_inverse(x: f32) -> f32 {
    if x <= 0. {
        return 0.;
    }
    let xx = f32::powf(x, 7.460772656268214e-03);
    let rs = 1e4
        * f32::powf(
            (0.8359375 - xx) / (18.6875 * xx - 18.8515625),
            6.277394636015326,
        );
    if rs.is_nan() {
        return 0.;
    }
    rs
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
/// Represents Jzazbz
pub struct Jzazbz {
    /// Jz(lightness) generally expects to be between [0;1]
    pub jz: f32,
    /// Az generally expects to be between [-0.5;0.5]
    pub az: f32,
    /// Bz generally expects to be between [-0.5;0.5]
    pub bz: f32,
    /// Display luminance, default 200 nits
    pub display_luminance: f32,
}

impl Jzazbz {
    /// Constructs new instance
    #[inline]
    pub fn new(jz: f32, az: f32, bz: f32) -> Jzazbz {
        Jzazbz {
            jz,
            az,
            bz,
            display_luminance: 200f32,
        }
    }

    /// Constructs new instance
    #[inline]
    pub fn new_with_luminance(jz: f32, az: f32, bz: f32, display_luminance: f32) -> Jzazbz {
        Jzazbz {
            jz,
            az,
            bz,
            display_luminance,
        }
    }

    #[inline]
    pub fn from_xyz(xyz: Xyz) -> Jzazbz {
        Self::from_xyz_with_display_luminance(xyz, 200f32)
    }

    #[inline]
    pub fn from_xyz_with_display_luminance(xyz: Xyz, display_luminance: f32) -> Jzazbz {
        let abs_xyz = xyz.to_absolute_luminance(display_luminance);
        let lp = perceptual_quantizer(
            0.674207838 * abs_xyz.x + 0.382799340 * abs_xyz.y - 0.047570458 * abs_xyz.z,
        );
        let mp = perceptual_quantizer(
            0.149284160 * abs_xyz.x + 0.739628340 * abs_xyz.y + 0.083327300 * abs_xyz.z,
        );
        let sp = perceptual_quantizer(
            0.070941080 * abs_xyz.x + 0.174768000 * abs_xyz.y + 0.670970020 * abs_xyz.z,
        );
        let iz = 0.5 * (lp + mp);
        let az = 3.524000 * lp - 4.066708 * mp + 0.542708 * sp;
        let bz = 0.199076 * lp + 1.096799 * mp - 1.295875 * sp;
        let jz = (0.44 * iz) / (1. - 0.56 * iz) - 1.6295499532821566e-11;
        Jzazbz::new_with_luminance(jz, az, bz, display_luminance)
    }

    /// Converts Rgb to Jzazbz
    /// Here is display luminance always considered as 200 nits
    ///
    /// # Arguments
    /// `transfer_function` - transfer function into linear color space and it's inverse
    #[inline]
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> Jzazbz {
        let xyz = rgb.to_xyz(&SRGB_TO_XYZ_D65, transfer_function);
        Self::from_xyz_with_display_luminance(xyz, 200.)
    }

    /// Converts Rgb to Jzazbz
    ///
    /// # Arguments
    /// `transfer_function` - transfer function into linear color space and it's inverse
    /// `display_luminance` - display luminance
    #[inline]
    pub fn from_rgb_with_luminance(
        rgb: Rgb<u8>,
        display_luminance: f32,
        transfer_function: TransferFunction,
    ) -> Jzazbz {
        let xyz = rgb.to_xyz(&SRGB_TO_XYZ_D65, transfer_function);
        Self::from_xyz_with_display_luminance(xyz, display_luminance)
    }

    /// Converts Jzazbz to *Xyz*
    #[inline]
    pub fn to_xyz(&self) -> Xyz {
        let jz = self.jz + 1.6295499532821566e-11;

        let iz = jz / (0.44f32 + 0.56f32 * jz);
        let l = perceptual_quantizer_inverse(
            iz + 1.386050432715393e-1 * self.az + 5.804731615611869e-2 * self.bz,
        );
        let m = perceptual_quantizer_inverse(
            iz - 1.386050432715393e-1 * self.az - 5.804731615611891e-2 * self.bz,
        );
        let s = perceptual_quantizer_inverse(
            iz - 9.601924202631895e-2 * self.az - 8.118918960560390e-1 * self.bz,
        );
        let x = 1.661373055774069e+00 * l - 9.145230923250668e-01 * m + 2.313620767186147e-01 * s;
        let y = -3.250758740427037e-01 * l + 1.571847038366936e+00 * m - 2.182538318672940e-01 * s;
        let z = -9.098281098284756e-02 * l - 3.127282905230740e-01 * m + 1.522766561305260e+00 * s;
        Xyz::new(x, y, z).to_relative_luminance(self.display_luminance)
    }

    /// Converts to Linear RGB
    #[inline]
    pub fn to_linear_rgb(&self) -> Rgb<f32> {
        let xyz = self.to_xyz();
        xyz.to_linear_rgb(&XYZ_TO_SRGB_D65)
    }

    /// Converts Linear to RGB with requested transfer function
    #[inline]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let linear_rgb = self.to_linear_rgb().gamma(transfer_function);
        linear_rgb.to_u8()
    }

    /// Converts into *Jzczhz*
    #[inline]
    pub fn to_jzczhz(&self) -> Jzczhz {
        Jzczhz::from_jzazbz(*self)
    }
}

impl EuclideanDistance for Jzazbz {
    #[inline]
    fn euclidean_distance(&self, other: Self) -> f32 {
        let djz = self.jz - other.jz;
        let daz = self.az - other.az;
        let dbz = self.bz - other.bz;
        (djz * djz + daz * daz + dbz * dbz).sqrt()
    }
}

impl TaxicabDistance for Jzazbz {
    #[inline]
    fn taxicab_distance(&self, other: Self) -> f32 {
        let djz = self.jz - other.jz;
        let daz = self.az - other.az;
        let dbz = self.bz - other.bz;
        djz.abs() + daz.abs() + dbz.abs()
    }
}

impl Index<usize> for Jzazbz {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.jz,
            1 => &self.az,
            2 => &self.bz,
            _ => panic!("Index out of bounds for Jzazbz"),
        }
    }
}

impl IndexMut<usize> for Jzazbz {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.jz,
            1 => &mut self.az,
            2 => &mut self.bz,
            _ => panic!("Index out of bounds for Jzazbz"),
        }
    }
}

impl Add<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz + rhs, self.az + rhs, self.bz + rhs)
    }
}

impl Sub<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz - rhs, self.az - rhs, self.bz - rhs)
    }
}

impl Mul<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz * rhs, self.az * rhs, self.bz * rhs)
    }
}

impl Div<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz / rhs, self.az / rhs, self.bz / rhs)
    }
}

impl Add<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn add(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(self.jz + rhs.jz, self.az + rhs.az, self.bz + rhs.bz)
    }
}

impl Sub<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn sub(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(self.jz - rhs.jz, self.az - rhs.az, self.bz - rhs.bz)
    }
}

impl Mul<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn mul(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(self.jz * rhs.jz, self.az * rhs.az, self.bz * rhs.bz)
    }
}

impl Div<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn div(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(self.jz / rhs.jz, self.az / rhs.az, self.bz / rhs.bz)
    }
}

impl AddAssign<Jzazbz> for Jzazbz {
    #[inline]
    fn add_assign(&mut self, rhs: Jzazbz) {
        self.jz += rhs.jz;
        self.az += rhs.az;
        self.bz += rhs.bz;
    }
}

impl SubAssign<Jzazbz> for Jzazbz {
    #[inline]
    fn sub_assign(&mut self, rhs: Jzazbz) {
        self.jz -= rhs.jz;
        self.az -= rhs.az;
        self.bz -= rhs.bz;
    }
}

impl MulAssign<Jzazbz> for Jzazbz {
    #[inline]
    fn mul_assign(&mut self, rhs: Jzazbz) {
        self.jz *= rhs.jz;
        self.az *= rhs.az;
        self.bz *= rhs.bz;
    }
}

impl DivAssign<Jzazbz> for Jzazbz {
    #[inline]
    fn div_assign(&mut self, rhs: Jzazbz) {
        self.jz /= rhs.jz;
        self.az /= rhs.az;
        self.bz /= rhs.bz;
    }
}

impl AddAssign<f32> for Jzazbz {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.jz += rhs;
        self.az += rhs;
        self.bz += rhs;
    }
}

impl SubAssign<f32> for Jzazbz {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.jz -= rhs;
        self.az -= rhs;
        self.bz -= rhs;
    }
}

impl MulAssign<f32> for Jzazbz {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.jz *= rhs;
        self.az *= rhs;
        self.bz *= rhs;
    }
}

impl DivAssign<f32> for Jzazbz {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.jz /= rhs;
        self.az /= rhs;
        self.bz /= rhs;
    }
}

impl Neg for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn neg(self) -> Self::Output {
        Jzazbz::new(-self.jz, -self.az, -self.bz)
    }
}

impl Jzazbz {
    #[inline]
    pub fn sqrt(&self) -> Jzazbz {
        Jzazbz::new(
            if self.jz < 0. { 0. } else { self.jz.sqrt() },
            if self.az < 0. { 0. } else { self.az.sqrt() },
            if self.bz < 0. { 0. } else { self.bz.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Jzazbz {
        Jzazbz::new(self.jz.cbrt(), self.az.cbrt(), self.bz.cbrt())
    }
}

impl Pow<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz.powf(rhs), self.az.powf(rhs), self.bz.powf(rhs))
    }
}

impl Pow<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn pow(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(
            self.jz.powf(rhs.jz),
            self.az.powf(self.az),
            self.bz.powf(self.bz),
        )
    }
}
