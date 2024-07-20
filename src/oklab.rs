/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{
    srgb_from_linear, srgb_to_linear, EuclideanDistance, Rgb, TaxicabDistance, TransferFunction,
};
use erydanos::ehypot3f;

#[derive(Copy, Clone)]
/// Struct that represent *Oklab* colorspace
pub struct Oklab {
    /// All values in Oklab intended to be normalized [0;1]
    pub l: f32,
    /// All values in Oklab intended to be normalized [0;1]
    pub a: f32,
    /// All values in Oklab intended to be normalized [0;1]
    pub b: f32,
}

impl Oklab {
    #[inline]
    pub fn new(l: f32, a: f32, b: f32) -> Oklab {
        Oklab { l, a, b }
    }

    #[inline]
    /// Converts from RGB to *Oklab* using *sRGB* transfer function
    pub fn from_srgb(rgb: Rgb<u8>) -> Oklab {
        let rgb_float = rgb.to_rgb_f32();
        let linearized = rgb_float.apply(srgb_to_linear);
        Self::linear_rgb_to_oklab(linearized)
    }

    #[inline]
    /// Converts from RGB to *Oklab* using provided transfer function
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> Oklab {
        let transfer = transfer_function.get_linearize_function();
        let rgb_float = rgb.to_rgb_f32();
        let linearized = rgb_float.apply(transfer);
        Self::linear_rgb_to_oklab(linearized)
    }

    #[inline]
    /// Converts to RGB using *sRGB* transfer function
    pub fn to_srgb(&self) -> Rgb<u8> {
        let linear_rgb = self.to_linear_srgb();
        let transferred = linear_rgb.apply(srgb_from_linear);
        transferred.to_u8()
    }

    #[inline]
    /// Converts to RGB using provided transfer function
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let linear_rgb = self.to_linear_srgb();
        let transfer = transfer_function.get_gamma_function();
        let transferred = linear_rgb.apply(transfer);
        transferred.to_u8()
    }

    #[inline]
    /// Converts to RGB using *sRGB* transfer function
    pub fn to_srgb_f32(&self) -> Rgb<f32> {
        let linear_rgb = self.to_linear_srgb();
        linear_rgb.apply(srgb_from_linear)
    }

    #[inline]
    /// Converts to RGB using provided transfer function
    pub fn to_rgb_f32(&self, transfer_function: TransferFunction) -> Rgb<f32> {
        let linear_rgb = self.to_linear_srgb();
        let transfer = transfer_function.get_gamma_function();
        linear_rgb.apply(transfer)
    }

    #[inline]
    fn linear_rgb_to_oklab(rgb: Rgb<f32>) -> Oklab {
        let l = 0.4122214708f32 * rgb.r + 0.5363325363f32 * rgb.g + 0.0514459929f32 * rgb.b;
        let m = 0.2119034982f32 * rgb.r + 0.6806995451f32 * rgb.g + 0.1073969566f32 * rgb.b;
        let s = 0.0883024619f32 * rgb.r + 0.2817188376f32 * rgb.g + 0.6299787005f32 * rgb.b;

        let l_ = l.cbrt();
        let m_ = m.cbrt();
        let s_ = s.cbrt();

        return Oklab {
            l: 0.2104542553f32 * l_ + 0.7936177850f32 * m_ - 0.0040720468f32 * s_,
            a: 1.9779984951f32 * l_ - 2.4285922050f32 * m_ + 0.4505937099f32 * s_,
            b: 0.0259040371f32 * l_ + 0.7827717662f32 * m_ - 0.8086757660f32 * s_,
        };
    }

    #[inline]
    /// Converts to linear RGB
    pub fn to_linear_srgb(&self) -> Rgb<f32> {
        let l_ = self.l + 0.3963377774f32 * self.a + 0.2158037573f32 * self.b;
        let m_ = self.l - 0.1055613458f32 * self.a - 0.0638541728f32 * self.b;
        let s_ = self.l - 0.0894841775f32 * self.a - 1.2914855480f32 * self.b;

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        return Rgb::<f32>::new(
            4.0767416621f32 * l - 3.3077115913f32 * m + 0.2309699292f32 * s,
            -1.2684380046f32 * l + 2.6097574011f32 * m - 0.3413193965f32 * s,
            -0.0041960863f32 * l - 0.7034186147f32 * m + 1.7076147010f32 * s,
        );
    }
}

impl EuclideanDistance for Oklab {
    fn euclidean_distance(&self, other: Self) -> f32 {
        ehypot3f(self.l - other.l, self.a - other.a, self.b - other.b)
    }
}

impl TaxicabDistance for Oklab {
    fn taxicab_distance(&self, other: Self) -> f32 {
        (self.a - other.a).hypot(self.b - other.b) + (self.l - other.l).abs()
    }
}
