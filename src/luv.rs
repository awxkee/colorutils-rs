//! # Luv
/// Struct representing a color in CIALuv, a.k.a. L\*u\*v\*, color space
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

/// Struct representing a color in cylindrical CIELCh(uv) color space
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

pub(crate) const LUV_WHITE_U_PRIME: f32 =
    4.0f32 * D65_XYZ[1] / (D65_XYZ[0] + 15.0 * D65_XYZ[1] + 3.0 * D65_XYZ[2]);
pub(crate) const LUV_WHITE_V_PRIME: f32 =
    9.0f32 * D65_XYZ[1] / (D65_XYZ[0] + 15.0 * D65_XYZ[1] + 3.0 * D65_XYZ[2]);

pub(crate) const LUV_CUTOFF_FORWARD_Y: f32 = (6f32 / 29f32) * (6f32 / 29f32) * (6f32 / 29f32);
pub(crate) const LUV_MULTIPLIER_FORWARD_Y: f32 = (29f32 / 3f32) * (29f32 / 3f32) * (29f32 / 3f32);
pub(crate) const LUV_MULTIPLIER_INVERSE_Y: f32 = (3f32 / 29f32) * (3f32 / 29f32) * (3f32 / 29f32);
impl Luv {
    pub fn from_rgb(rgb: &Rgb<u8>) -> Self {
        let xyz = Xyz::from_srgb(rgb);
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

    pub fn from_rgba(rgba: &Rgba<u8>) -> Self {
        Luv::from_rgb(&rgba.to_rgb())
    }

    #[allow(dead_code)]
    pub fn to_rgb(&self) -> Rgb<u8> {
        if self.l <= 0f32 {
            return Xyz::new(0f32, 0f32, 0f32).to_srgb();
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

        Xyz::new(x, y, z).to_srgb()
    }

    pub fn new(l: f32, u: f32, v: f32) -> Luv {
        Luv { l, u, v }
    }
}

impl LCh {
    pub fn from_rgb(rgb: &Rgb<u8>) -> Self {
        LCh::from_luv(Luv::from_rgb(rgb))
    }

    pub fn from_rgba(rgba: &Rgba<u8>) -> Self {
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

    pub fn to_rgb(&self) -> Rgb<u8> {
        self.to_luv().to_rgb()
    }

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
