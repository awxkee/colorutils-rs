/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::gamma_curves::TransferFunction;
use crate::rgb::Rgb;
use crate::{EuclideanDistance, Jzazbz, SRGB_TO_XYZ_D65, XYZ_TO_SRGB_D65};
use erydanos::Euclidean3DDistance;

/// A CIE 1931 XYZ color.
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
        x.max(0f32).min(95.047f32)
    }

    #[inline]
    pub fn saturate_y(y: f32) -> f32 {
        y.max(0f32).min(100f32)
    }

    #[inline]
    pub fn saturate_z(z: f32) -> f32 {
        z.max(0f32).min(108.883f32)
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
    pub fn from_srgb(rgb: &Rgb<u8>) -> Self {
        Xyz::from_rgb(rgb, &SRGB_TO_XYZ_D65, TransferFunction::Srgb)
    }

    /// This function converts from non-linear RGB components to XYZ
    /// # Arguments
    /// * `matrix` - Transformation matrix from RGB to XYZ, for example `SRGB_TO_XYZ_D65`
    /// * `transfer_function` - Transfer functions for current colorspace
    #[inline]
    pub fn from_rgb(
        rgb: &Rgb<u8>,
        matrix: &[[f32; 3]; 3],
        transfer_function: TransferFunction,
    ) -> Self {
        let linear_function = transfer_function.get_linearize_function();
        unsafe {
            let r = linear_function(rgb.r as f32 * XYZ_SCALE_U8);
            let g = linear_function(rgb.g as f32 * XYZ_SCALE_U8);
            let b = linear_function(rgb.b as f32 * XYZ_SCALE_U8);
            Self::new(
                (*(*matrix.get_unchecked(0)).get_unchecked(0)) * r
                    + (*(*matrix.get_unchecked(0)).get_unchecked(1)) * g
                    + (*(*matrix.get_unchecked(0)).get_unchecked(2)) * b,
                (*(*matrix.get_unchecked(1)).get_unchecked(0)) * r
                    + (*(*matrix.get_unchecked(1)).get_unchecked(1)) * g
                    + (*(*matrix.get_unchecked(1)).get_unchecked(2)) * b,
                (*(*matrix.get_unchecked(2)).get_unchecked(0)) * r
                    + (*(*matrix.get_unchecked(2)).get_unchecked(1)) * g
                    + (*(*matrix.get_unchecked(2)).get_unchecked(2)) * b,
            )
        }
    }

    /// This function converts from non-linear RGB components to XYZ
    /// # Arguments
    /// * `matrix` - Transformation matrix from RGB to XYZ, for example `SRGB_TO_XYZ_D65`
    /// * `transfer_function` - Transfer functions for current colorspace
    #[inline]
    pub fn from_linear_rgb(rgb: &Rgb<f32>, matrix: &[[f32; 3]; 3]) -> Self {
        unsafe {
            Self::new(
                (*(*matrix.get_unchecked(0)).get_unchecked(0)) * rgb.r
                    + (*(*matrix.get_unchecked(0)).get_unchecked(1)) * rgb.g
                    + (*(*matrix.get_unchecked(0)).get_unchecked(2)) * rgb.b,
                (*(*matrix.get_unchecked(1)).get_unchecked(0)) * rgb.r
                    + (*(*matrix.get_unchecked(1)).get_unchecked(1)) * rgb.g
                    + (*(*matrix.get_unchecked(1)).get_unchecked(2)) * rgb.b,
                (*(*matrix.get_unchecked(2)).get_unchecked(0)) * rgb.r
                    + (*(*matrix.get_unchecked(2)).get_unchecked(1)) * rgb.g
                    + (*(*matrix.get_unchecked(2)).get_unchecked(2)) * rgb.b,
            )
        }
    }

    pub fn scaled(&self) -> (f32, f32, f32) {
        (self.x * 100f32, self.y * 100f32, self.z * 100f32)
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
        let gamma_function = transfer_function.get_gamma_function();
        let x = self.x;
        let y = self.y;
        let z = self.z;
        unsafe {
            let r = x * (*(*matrix.get_unchecked(0)).get_unchecked(0))
                + y * (*(*matrix.get_unchecked(0)).get_unchecked(1))
                + z * (*(*matrix.get_unchecked(0)).get_unchecked(2));
            let g = x * (*(*matrix.get_unchecked(1)).get_unchecked(0))
                + y * (*(*matrix.get_unchecked(1)).get_unchecked(1))
                + z * (*(*matrix.get_unchecked(1)).get_unchecked(2));
            let b = x * (*(*matrix.get_unchecked(2)).get_unchecked(0))
                + y * (*(*matrix.get_unchecked(2)).get_unchecked(1))
                + z * (*(*matrix.get_unchecked(2)).get_unchecked(2));
            let r = 255f32 * gamma_function(r);
            let g = 255f32 * gamma_function(g);
            let b = 255f32 * gamma_function(b);
            Rgb::new(r as u8, g as u8, b as u8)
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
            let r = x * (*(*matrix.get_unchecked(0)).get_unchecked(0))
                + y * (*(*matrix.get_unchecked(0)).get_unchecked(1))
                + z * (*(*matrix.get_unchecked(0)).get_unchecked(2));
            let g = x * (*(*matrix.get_unchecked(1)).get_unchecked(0))
                + y * (*(*matrix.get_unchecked(1)).get_unchecked(1))
                + z * (*(*matrix.get_unchecked(1)).get_unchecked(2));
            let b = x * (*(*matrix.get_unchecked(2)).get_unchecked(0))
                + y * (*(*matrix.get_unchecked(2)).get_unchecked(1))
                + z * (*(*matrix.get_unchecked(2)).get_unchecked(2));
            Rgb::<f32>::new(r, g, b)
        }
    }
}

impl EuclideanDistance for Xyz {
    fn euclidean_distance(&self, other: Xyz) -> f32 {
        (self.x - other.x).hypot3(self.y - other.y, self.z - other.z)
    }
}
