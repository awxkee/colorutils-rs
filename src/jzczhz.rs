/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{EuclideanDistance, Jzazbz, Rgb, TransferFunction, Xyz};
use erydanos::{eatan2f, ehypot3f, ehypotf, Cosine, Sine};

/// Represents Jzazbz in polar coordinates as Jzczhz
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

    /// Converts Jzczhz to *Xyz*
    #[inline]
    pub fn to_xyz(&self) -> Xyz {
        let jzazbz = self.to_jzazbz();
        jzazbz.to_xyz()
    }

    /// Converts *Xyz* to *Jzczhz*
    #[inline]
    pub fn from_xyz(xyz: Xyz) -> Jzczhz {
        let jzazbz = Jzazbz::from_xyz(xyz);
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
    fn euclidean_distance(&self, other: Self) -> f32 {
        ehypot3f(self.jz - other.jz, self.hz - other.hz, self.cz - other.cz)
    }
}
