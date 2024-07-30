/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{Rgb, TransferFunction, Xyz, SRGB_TO_XYZ_D65, XYZ_TO_SRGB_D65};
use erydanos::{Exponential, Logarithmic};

/// Represents l-alpha-beta colorspace
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct LAlphaBeta {
    pub l: f32,
    pub alpha: f32,
    pub beta: f32,
}

impl LAlphaBeta {
    #[inline]
    /// Creates new instance
    pub fn new(l: f32, alpha: f32, beta: f32) -> LAlphaBeta {
        LAlphaBeta { l, alpha, beta }
    }

    /// Converts RGB to l-alpha-beta
    #[inline]
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> LAlphaBeta {
        let xyz = rgb.to_xyz(&SRGB_TO_XYZ_D65, transfer_function);
        LAlphaBeta::from_xyz(xyz)
    }

    /// Converts XYZ to l-alpha-beta
    #[inline]
    pub fn from_xyz(xyz: Xyz) -> LAlphaBeta {
        let l_lms = 0.3897 * xyz.x + 0.6890 * xyz.y - 0.0787 * xyz.z;
        let m = -0.2298 * xyz.x + 1.1834 * xyz.y + 0.0464 * xyz.z;
        let s = xyz.z;
        let lp = if l_lms > 0. { l_lms.eln() } else { 0. };
        let mp = if m > 0. { m.eln() } else { 0. };
        let sp = if s > 0. { s.eln() } else { 0. };
        let l = 0.5774 * lp + 0.5774 * mp + 0.5774 * sp;
        let alpha = 0.4082 * lp + 0.4082 * mp - 0.8165 * sp;
        let beta = 1.4142 * lp - 1.4142 * mp;
        LAlphaBeta::new(l, alpha, beta)
    }

    /// Converts l-alpha-beta to XYZ
    #[inline]
    pub fn to_xyz(&self) -> Xyz {
        let l_a = self.l * 0.577324;
        let s_a = 0.408263 * self.alpha;
        let p_b = 0.353557 * self.beta;
        let lp = l_a + s_a + p_b;
        let mp = l_a + s_a - p_b;
        let sp = self.l * 0.577253 - 0.816526 * self.alpha;
        let l = lp.eexp();
        let m = mp.eexp();
        let s = sp.eexp();
        let x = 1.91024 * l - 1.11218 * m + 0.201941 * s;
        let y = 0.370942 * l + 0.62905 * m + 5.13315e-6 * s;
        let z = s;
        Xyz::new(x, y, z)
    }

    /// Converts l-alpha-beta to RGB
    #[inline]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let xyz = self.to_xyz();
        xyz.to_rgb(&XYZ_TO_SRGB_D65, transfer_function)
    }
}
