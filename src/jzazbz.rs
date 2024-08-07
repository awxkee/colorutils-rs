/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{
    EuclideanDistance, Jzczhz, Rgb, TransferFunction, Xyz, SRGB_TO_XYZ_D65, XYZ_TO_SRGB_D65,
};
use erydanos::ehypot3f;

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

    /// Converts to RGB with requested transfer function
    #[inline]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let linear_rgb = self
            .to_linear_rgb()
            .apply(transfer_function.get_gamma_function());
        linear_rgb.to_u8()
    }

    /// Converts into *Jzczhz*
    #[inline]
    pub fn to_jzczhz(&self) -> Jzczhz {
        Jzczhz::from_jzazbz(*self)
    }
}

impl EuclideanDistance for Jzazbz {
    fn euclidean_distance(&self, other: Self) -> f32 {
        ehypot3f(self.jz - other.jz, self.az - other.az, self.bz - other.bz)
    }
}
