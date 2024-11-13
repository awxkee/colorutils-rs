/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::gamma_curves::TransferFunction;
use crate::rgb::Rgb;
use crate::utils::mlaf;
use crate::{EuclideanDistance, Jzazbz, SRGB_TO_XYZ_D65, XYZ_TO_SRGB_D65};
use erydanos::Euclidean3DDistance;
use num_traits::Pow;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A CIE 1931 XYZ color.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Xyz {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Xyz {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn saturate_x(x: f32) -> f32 {
        #[allow(clippy::manual_clamp)]
        x.max(0f32).min(95.047f32)
    }

    #[inline]
    pub fn saturate_y(y: f32) -> f32 {
        #[allow(clippy::manual_clamp)]
        y.max(0f32).min(100f32)
    }

    #[inline]
    pub fn saturate_z(z: f32) -> f32 {
        #[allow(clippy::manual_clamp)]
        z.max(0f32).min(108.883f32)
    }

    #[inline]
    pub fn scale(&self, by: f32) -> Xyz {
        Xyz {
            x: self.x * by,
            y: self.y * by,
            z: self.z * by,
        }
    }

    /// Scales XYZ to absolute luminance against display
    #[inline]
    pub fn to_absolute_luminance(&self, display_nits: f32) -> Xyz {
        let multiplier = display_nits;
        Xyz::new(
            multiplier * self.x,
            multiplier * self.y,
            multiplier * self.z,
        )
    }

    /// Scales XYZ to absolute luminance against display
    #[inline]
    pub fn to_relative_luminance(&self, display_nits: f32) -> Xyz {
        let multiplier = 1. / display_nits;
        Xyz::new(
            multiplier * self.x,
            multiplier * self.y,
            multiplier * self.z,
        )
    }
}

static XYZ_SCALE_U8: f32 = 1f32 / 255f32;

/// This class avoid to scale by 100 for a reason: in common there are no need to scale by 100 in digital image processing,
/// Normalized values are speeding up computing.
/// if you need this multiply by yourself or use `scaled`
impl Xyz {
    /// Converts into *Xyz*
    #[inline]
    pub fn to_jzazbz(&self) -> Jzazbz {
        Jzazbz::from_xyz(*self)
    }

    /// This functions always use sRGB transfer function and Rec.601 primaries with D65 White point
    #[inline]
    pub fn from_srgb(rgb: Rgb<u8>) -> Self {
        Xyz::from_rgb(rgb, &SRGB_TO_XYZ_D65, TransferFunction::Srgb)
    }

    /// This function converts from non-linear RGB components to XYZ
    /// # Arguments
    /// * `matrix` - Transformation matrix from RGB to XYZ, for example `SRGB_TO_XYZ_D65`
    /// * `transfer_function` - Transfer functions for current colorspace
    #[inline]
    pub fn from_rgb(
        rgb: Rgb<u8>,
        matrix: &[[f32; 3]; 3],
        transfer_function: TransferFunction,
    ) -> Self {
        unsafe {
            let r = transfer_function.linearize(rgb.r as f32 * XYZ_SCALE_U8);
            let g = transfer_function.linearize(rgb.g as f32 * XYZ_SCALE_U8);
            let b = transfer_function.linearize(rgb.b as f32 * XYZ_SCALE_U8);
            Self::new(
                mlaf(
                    mlaf(
                        (*(*matrix.get_unchecked(0)).get_unchecked(0)) * r,
                        *(*matrix.get_unchecked(0)).get_unchecked(1),
                        g,
                    ),
                    *(*matrix.get_unchecked(0)).get_unchecked(2),
                    b,
                ),
                mlaf(
                    mlaf(
                        (*(*matrix.get_unchecked(1)).get_unchecked(0)) * r,
                        *(*matrix.get_unchecked(1)).get_unchecked(1),
                        g,
                    ),
                    *(*matrix.get_unchecked(1)).get_unchecked(2),
                    b,
                ),
                mlaf(
                    mlaf(
                        (*(*matrix.get_unchecked(2)).get_unchecked(0)) * r,
                        *(*matrix.get_unchecked(2)).get_unchecked(1),
                        g,
                    ),
                    *(*matrix.get_unchecked(2)).get_unchecked(2),
                    b,
                ),
            )
        }
    }

    /// This function converts from non-linear RGB components to XYZ
    /// # Arguments
    /// * `matrix` - Transformation matrix from RGB to XYZ, for example `SRGB_TO_XYZ_D65`
    /// * `transfer_function` - Transfer functions for current colorspace
    #[inline]
    pub fn from_linear_rgb(rgb: Rgb<f32>, matrix: &[[f32; 3]; 3]) -> Self {
        unsafe {
            Self::new(
                mlaf(
                    mlaf(
                        (*(*matrix.get_unchecked(0)).get_unchecked(0)) * rgb.r,
                        *(*matrix.get_unchecked(0)).get_unchecked(1),
                        rgb.g,
                    ),
                    *(*matrix.get_unchecked(0)).get_unchecked(2),
                    rgb.b,
                ),
                mlaf(
                    mlaf(
                        (*(*matrix.get_unchecked(1)).get_unchecked(0)) * rgb.r,
                        *(*matrix.get_unchecked(1)).get_unchecked(1),
                        rgb.g,
                    ),
                    *(*matrix.get_unchecked(1)).get_unchecked(2),
                    rgb.b,
                ),
                mlaf(
                    mlaf(
                        (*(*matrix.get_unchecked(2)).get_unchecked(0)) * rgb.r,
                        *(*matrix.get_unchecked(2)).get_unchecked(1),
                        rgb.g,
                    ),
                    *(*matrix.get_unchecked(2)).get_unchecked(2),
                    rgb.b,
                ),
            )
        }
    }

    #[inline]
    pub fn scaled(&self) -> (f32, f32, f32) {
        (self.x * 100f32, self.y * 100f32, self.z * 100f32)
    }

    #[inline]
    pub fn scaled_by(&self, by: f32) -> Xyz {
        Xyz::new(self.x * by, self.y * by, self.z * by)
    }
}

impl Xyz {
    /// This functions always use sRGB transfer function and Rec.601 primaries with D65 White point
    pub fn to_srgb(&self) -> Rgb<u8> {
        self.to_rgb(&XYZ_TO_SRGB_D65, TransferFunction::Srgb)
    }

    /// This functions always use sRGB transfer function and Rec.601 primaries with D65 White point
    /// # Arguments
    /// * `matrix` - Transformation matrix from RGB to XYZ, for example `SRGB_TO_XYZ_D65`
    /// * `transfer_function` - Transfer functions for current colorspace
    #[inline]
    pub fn to_rgb(&self, matrix: &[[f32; 3]; 3], transfer_function: TransferFunction) -> Rgb<u8> {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        unsafe {
            let r = mlaf(
                mlaf(
                    x * (*(*matrix.get_unchecked(0)).get_unchecked(0)),
                    y,
                    *(*matrix.get_unchecked(0)).get_unchecked(1),
                ),
                z,
                *(*matrix.get_unchecked(0)).get_unchecked(2),
            );
            let g = mlaf(
                mlaf(
                    x * (*(*matrix.get_unchecked(1)).get_unchecked(0)),
                    y,
                    *(*matrix.get_unchecked(1)).get_unchecked(1),
                ),
                z,
                *(*matrix.get_unchecked(1)).get_unchecked(2),
            );
            let b = mlaf(
                mlaf(
                    x * (*(*matrix.get_unchecked(2)).get_unchecked(0)),
                    y,
                    *(*matrix.get_unchecked(2)).get_unchecked(1),
                ),
                z,
                *(*matrix.get_unchecked(2)).get_unchecked(2),
            );
            Rgb::<f32>::new(
                transfer_function.gamma(r),
                transfer_function.gamma(g),
                transfer_function.gamma(b),
            )
            .to_u8()
        }
    }

    /// This function converts XYZ to linear RGB
    /// # Arguments
    /// * `matrix` - Transformation matrix from RGB to XYZ, for example `SRGB_TO_XYZ_D65`
    #[inline]
    pub fn to_linear_rgb(&self, matrix: &[[f32; 3]; 3]) -> Rgb<f32> {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        unsafe {
            let r = mlaf(
                mlaf(
                    x * (*(*matrix.get_unchecked(0)).get_unchecked(0)),
                    y,
                    *(*matrix.get_unchecked(0)).get_unchecked(1),
                ),
                z,
                *(*matrix.get_unchecked(0)).get_unchecked(2),
            );
            let g = mlaf(
                mlaf(
                    x * (*(*matrix.get_unchecked(1)).get_unchecked(0)),
                    y,
                    *(*matrix.get_unchecked(1)).get_unchecked(1),
                ),
                z,
                *(*matrix.get_unchecked(1)).get_unchecked(2),
            );
            let b = mlaf(
                mlaf(
                    x * (*(*matrix.get_unchecked(2)).get_unchecked(0)),
                    y,
                    *(*matrix.get_unchecked(2)).get_unchecked(1),
                ),
                z,
                *(*matrix.get_unchecked(2)).get_unchecked(2),
            );
            Rgb::<f32>::new(r, g, b)
        }
    }
}

impl EuclideanDistance for Xyz {
    fn euclidean_distance(&self, other: Xyz) -> f32 {
        (self.x - other.x).hypot3(self.y - other.y, self.z - other.z)
    }
}

impl Index<usize> for Xyz {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Xyz"),
        }
    }
}

impl IndexMut<usize> for Xyz {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Xyz"),
        }
    }
}

impl Add<Xyz> for Xyz {
    type Output = Xyz;

    #[inline]
    fn add(self, rhs: Self) -> Xyz {
        Xyz::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<f32> for Xyz {
    type Output = Xyz;

    #[inline]
    fn add(self, rhs: f32) -> Xyz {
        Xyz::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl AddAssign<Xyz> for Xyz {
    #[inline]
    fn add_assign(&mut self, rhs: Xyz) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl AddAssign<f32> for Xyz {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

impl Mul<f32> for Xyz {
    type Output = Xyz;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Xyz::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Xyz> for Xyz {
    type Output = Xyz;

    #[inline]
    fn mul(self, rhs: Xyz) -> Self::Output {
        Xyz::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl MulAssign<Xyz> for Xyz {
    #[inline]
    fn mul_assign(&mut self, rhs: Xyz) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl MulAssign<f32> for Xyz {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Sub<f32> for Xyz {
    type Output = Xyz;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Xyz::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl Sub<Xyz> for Xyz {
    type Output = Xyz;

    #[inline]
    fn sub(self, rhs: Xyz) -> Self::Output {
        Xyz::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign<f32> for Xyz {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
    }
}

impl SubAssign<Xyz> for Xyz {
    #[inline]
    fn sub_assign(&mut self, rhs: Xyz) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Div<f32> for Xyz {
    type Output = Xyz;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Xyz::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Div<Xyz> for Xyz {
    type Output = Xyz;

    #[inline]
    fn div(self, rhs: Xyz) -> Self::Output {
        Xyz::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl DivAssign<f32> for Xyz {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl DivAssign<Xyz> for Xyz {
    #[inline]
    fn div_assign(&mut self, rhs: Xyz) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl Neg for Xyz {
    type Output = Xyz;

    #[inline]
    fn neg(self) -> Self::Output {
        Xyz::new(-self.x, -self.y, -self.z)
    }
}

impl Xyz {
    #[inline]
    pub fn sqrt(&self) -> Xyz {
        Xyz::new(
            if self.x < 0. { 0. } else { self.x.sqrt() },
            if self.y < 0. { 0. } else { self.y.sqrt() },
            if self.z < 0. { 0. } else { self.z.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Xyz {
        Xyz::new(self.x.cbrt(), self.y.cbrt(), self.z.cbrt())
    }
}

impl Pow<f32> for Xyz {
    type Output = Xyz;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Xyz::new(self.x.powf(rhs), self.y.powf(rhs), self.z.powf(rhs))
    }
}

impl Pow<Xyz> for Xyz {
    type Output = Xyz;

    #[inline]
    fn pow(self, rhs: Xyz) -> Self::Output {
        Xyz::new(self.x.powf(rhs.x), self.y.powf(rhs.y), self.z.powf(rhs.z))
    }
}
