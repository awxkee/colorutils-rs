/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[inline]
/// Linear transfer function for sRGB
pub fn srgb_to_linear(gamma: f32) -> f32 {
    if gamma < 0f32 {
        0f32
    } else if gamma < 12.92f32 * 0.0030412825601275209f32 {
        gamma * (1f32 / 12.92f32)
    } else if gamma < 1.0f32 {
        ((gamma + 0.0550107189475866f32) / 1.0550107189475866f32).powf(2.4f32)
    } else {
        1.0f32
    }
}

#[inline]
/// Gamma transfer function for sRGB
pub fn srgb_from_linear(linear: f32) -> f32 {
    if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.0030412825601275209f32 {
        linear * 12.92f32
    } else if linear < 1.0f32 {
        1.0550107189475866f32 * linear.powf(1.0f32 / 2.4f32) - 0.0550107189475866f32
    } else {
        1.0f32
    }
}

#[inline]
/// Linear transfer function for Rec.709
pub fn rec709_to_linear(gamma: f32) -> f32 {
    if gamma < 0.0f32 {
        0.0f32
    } else if gamma < 4.5f32 * 0.018053968510807f32 {
        gamma * (1f32 / 4.5f32)
    } else if gamma < 1.0f32 {
        ((gamma + 0.09929682680944f32) / 1.09929682680944f32).powf(1.0f32 / 0.45f32)
    } else {
        1.0f32
    }
}

#[inline]
/// Gamma transfer function for Rec.709
pub fn rec709_from_linear(linear: f32) -> f32 {
    if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.018053968510807f32 {
        linear * 4.5f32
    } else if linear < 1.0f32 {
        1.09929682680944f32 * linear.powf(0.45f32) - 0.09929682680944f32
    } else {
        1.0f32
    }
}

#[inline]
/// Linear transfer function for Smpte 428
pub fn smpte428_to_linear(gamma: f32) -> f32 {
    const SCALE: f32 = 1. / 0.91655527974030934f32;
    gamma.max(0.).powf(2.6f32) * SCALE
}

#[inline]
/// Gamma transfer function for Smpte 428
pub fn smpte428_from_linear(linear: f32) -> f32 {
    const POWER_VALUE: f32 = 1.0f32 / 2.6f32;
    (0.91655527974030934f32 * linear.max(0.)).powf(POWER_VALUE)
}

#[inline(always)]
/// Pure gamma transfer function for gamma 2.2
pub fn pure_gamma_function(x: f32, gamma: f32) -> f32 {
    if x <= 0f32 {
        0f32
    } else if x >= 1f32 {
        return 1f32;
    } else {
        return x.powf(gamma);
    }
}

#[inline]
/// Pure gamma transfer function for gamma 2.2
pub fn gamma2p2_from_linear(linear: f32) -> f32 {
    pure_gamma_function(linear, 1f32 / 2.2f32)
}

#[inline]
/// Linear transfer function for gamma 2.2
pub fn gamma2p2_to_linear(gamma: f32) -> f32 {
    pure_gamma_function(gamma, 2.2f32)
}

#[inline]
/// Pure gamma transfer function for gamma 2.8
pub fn gamma2p8_from_linear(linear: f32) -> f32 {
    pure_gamma_function(linear, 1f32 / 2.8f32)
}

#[inline]
/// Linear transfer function for gamma 2.8
pub fn gamma2p8_to_linear(gamma: f32) -> f32 {
    pure_gamma_function(gamma, 2.8f32)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Declares transfer function for transfer components into a linear colorspace and its inverse
pub enum TransferFunction {
    /// sRGB Transfer function
    Srgb,
    /// Rec.709 Transfer function
    Rec709,
    /// Pure gamma 2.2 Transfer function
    Gamma2p2,
    /// Pure gamma 2.8 Transfer function
    Gamma2p8,
    /// Smpte 428 Transfer function
    Smpte428,
}

impl From<u8> for TransferFunction {
    #[inline]
    fn from(value: u8) -> Self {
        match value {
            0 => TransferFunction::Srgb,
            1 => TransferFunction::Rec709,
            2 => TransferFunction::Gamma2p2,
            3 => TransferFunction::Gamma2p8,
            4 => TransferFunction::Smpte428,
            _ => TransferFunction::Srgb,
        }
    }
}

impl TransferFunction {
    #[inline]
    pub fn linearize(&self, v: f32) -> f32 {
        match self {
            TransferFunction::Srgb => srgb_to_linear(v),
            TransferFunction::Rec709 => rec709_to_linear(v),
            TransferFunction::Gamma2p8 => gamma2p8_to_linear(v),
            TransferFunction::Gamma2p2 => gamma2p2_to_linear(v),
            TransferFunction::Smpte428 => smpte428_to_linear(v),
        }
    }

    #[inline]
    pub fn gamma(&self, v: f32) -> f32 {
        match self {
            TransferFunction::Srgb => srgb_from_linear(v),
            TransferFunction::Rec709 => rec709_from_linear(v),
            TransferFunction::Gamma2p2 => gamma2p2_from_linear(v),
            TransferFunction::Gamma2p8 => gamma2p8_from_linear(v),
            TransferFunction::Smpte428 => smpte428_to_linear(v),
        }
    }
}
