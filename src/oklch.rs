/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::{Oklab, Rgb, TransferFunction};
use erydanos::{eatan2f, ehypotf, Cosine, Sine};

/// Represents *Oklch* colorspace
#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub struct Oklch {
    /// Lightness
    pub l: f32,
    /// Chroma
    pub c: f32,
    /// Hue
    pub h: f32,
}

impl Oklch {
    /// Creates new instance
    #[inline]
    pub fn new(l: f32, c: f32, h: f32) -> Oklch {
        Oklch { l, c, h }
    }

    /// Converts *Rgb* into *Oklch*
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function into linear colorspace and its inverse
    #[inline]
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> Oklch {
        let oklab = rgb.to_oklab(transfer_function);
        Oklch::from_oklab(oklab)
    }

    /// Converts *Oklch* into *Rgb*
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function into linear colorspace and its inverse
    #[inline]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let oklab = self.to_oklab();
        oklab.to_rgb(transfer_function)
    }

    /// Converts *Oklab* to *Oklch*
    #[inline]
    pub fn from_oklab(oklab: Oklab) -> Oklch {
        let chroma = ehypotf(oklab.b, oklab.a);
        let hue = eatan2f(oklab.b, oklab.a);
        Oklch::new(oklab.l, chroma, hue)
    }

    /// Converts *Oklch* to *Oklab*
    #[inline]
    pub fn to_oklab(&self) -> Oklab {
        let l = self.l;
        let a = self.c * self.h.ecos();
        let b = self.c * self.h.esin();
        Oklab::new(l, a, b)
    }
}
