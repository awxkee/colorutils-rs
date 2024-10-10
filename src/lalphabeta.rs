/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{Rgb, TransferFunction, Xyz, SRGB_TO_XYZ_D65, XYZ_TO_SRGB_D65};
use std::ops::{Index, IndexMut, Neg};

/// Represents l-alpha-beta (lαβ) colorspace
#[repr(C)]
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

    #[inline]
    /// Converts linear [Rgb] to [LAlphaBeta] using [Xyz] matrix
    pub fn from_linear_rgb(rgb: Rgb<f32>, matrix: &[[f32; 3]; 3]) -> LAlphaBeta {
        let xyz = Xyz::from_linear_rgb(rgb, matrix);
        LAlphaBeta::from_xyz(xyz)
    }

    /// Converts XYZ to l-alpha-beta
    #[inline]
    pub fn from_xyz(xyz: Xyz) -> LAlphaBeta {
        let l_lms = 0.3897 * xyz.x + 0.6890 * xyz.y - 0.0787 * xyz.z;
        let m = -0.2298 * xyz.x + 1.1834 * xyz.y + 0.0464 * xyz.z;
        let s = xyz.z;
        let lp = if l_lms > 0. { l_lms.log10() } else { 0. };
        let mp = if m > 0. { m.log10() } else { 0. };
        let sp = if s > 0. { s.log10() } else { 0. };
        const ONE_F_SQRT3: f32 = 0.57735026918962576f32;
        const ONE_F_SQRT6: f32 = 0.40824829046386301f32;
        const TWO_F_SQRT6: f32 = 0.816496580927726032f32;
        const ONE_F_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let l = ONE_F_SQRT3 * lp + ONE_F_SQRT3 * mp + ONE_F_SQRT3 * sp;
        let alpha = ONE_F_SQRT6 * lp + ONE_F_SQRT6 * mp - TWO_F_SQRT6 * sp;
        let beta = ONE_F_SQRT2 * lp - ONE_F_SQRT2 * mp;
        LAlphaBeta::new(l, alpha, beta)
    }

    /// Converts l-alpha-beta to XYZ
    #[inline]
    pub fn to_xyz(&self) -> Xyz {
        const ONE_F_SQRT3: f32 = 0.57735026918962576f32;
        const ONE_F_SQRT6: f32 = 0.40824829046386301f32;
        const ONE_F_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        const TWO_F_SQRT6: f32 = 0.816496580927726032f32;
        let l_a = self.l * ONE_F_SQRT3;
        let s_a = ONE_F_SQRT6 * self.alpha;
        let p_b = ONE_F_SQRT2 * self.beta;
        let lp = l_a + s_a + p_b;
        let mp = l_a + s_a - p_b;
        let sp = l_a - TWO_F_SQRT6 * self.alpha;
        let l = if lp == 0. { 0. } else { 10f32.powf(lp) };
        let m = if mp == 0. { 0. } else { 10f32.powf(mp) };
        let s = if sp == 0. { 0. } else { 10f32.powf(sp) };
        let x = 1.91024 * l - 1.11218 * m + 0.201941 * s;
        let y = 0.370942 * l + 0.62905 * m + 5.13315e-6 * s;
        let z = s;
        Xyz::new(x, y, z)
    }

    /// Converts l-alpha-beta to [Rgb]
    #[inline]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let xyz = self.to_xyz();
        xyz.to_rgb(&XYZ_TO_SRGB_D65, transfer_function)
    }

    /// Converts l-alpha-beta to Linear [Rgb]
    #[inline]
    pub fn to_linear_rgb(&self, matrix: &[[f32; 3]; 3]) -> Rgb<f32> {
        let xyz = self.to_xyz();
        xyz.to_linear_rgb(matrix)
    }
}

impl Index<usize> for LAlphaBeta {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.l,
            1 => &self.alpha,
            2 => &self.beta,
            _ => panic!("Index out of bounds for LAlphaBeta"),
        }
    }
}

impl IndexMut<usize> for LAlphaBeta {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.l,
            1 => &mut self.alpha,
            2 => &mut self.beta,
            _ => panic!("Index out of bounds for LAlphaBeta"),
        }
    }
}

impl Neg for LAlphaBeta {
    type Output = LAlphaBeta;

    fn neg(self) -> Self::Output {
        LAlphaBeta::new(-self.l, -self.alpha, -self.beta)
    }
}
