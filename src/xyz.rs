use crate::gamma_curves::TransferFunction;
use crate::rgb::Rgb;
use crate::{SRGB_TO_XYZ_D65, XYZ_TO_SRGB_D65};

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
    /// This functions always use sRGB transfer function and Rec.601 primaries with D65 White point
    #[inline(always)]
    pub fn from_srgb(rgb: &Rgb<u8>) -> Self {
        Xyz::from_rgb(rgb, &SRGB_TO_XYZ_D65, TransferFunction::Srgb)
    }

    /// This function converts from non-linear RGB components to XYZ
    /// # Arguments
    /// * `matrix` - Transformation matrix from RGB to XYZ, for example `SRGB_TO_XYZ_D65`
    /// * `transfer_function` - Transfer functions for current colorspace
    #[inline(always)]
    pub fn from_rgb(
        rgb: &Rgb<u8>,
        matrix: &[[f32; 3]; 3],
        transfer_function: TransferFunction,
    ) -> Self {
        let linear_function = transfer_function.get_linearize_function();
        let r = linear_function(rgb.r as f32 * XYZ_SCALE_U8);
        let g = linear_function(rgb.g as f32 * XYZ_SCALE_U8);
        let b = linear_function(rgb.b as f32 * XYZ_SCALE_U8);
        Self::new(
            matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b,
            matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b,
            matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b,
        )
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
    #[inline(always)]
    pub fn to_rgb(&self, matrix: &[[f32; 3]; 3], transfer_function: TransferFunction) -> Rgb<u8> {
        let gamma_function = transfer_function.get_gamma_function();
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let r = x * matrix[0][0] + y * matrix[0][1] + z * matrix[0][2];
        let g = x * matrix[1][0] + y * matrix[1][1] + z * matrix[1][2];
        let b = x * matrix[2][0] + y * matrix[2][1] + z * matrix[2][2];
        let r = 255f32 * gamma_function(r);
        let g = 255f32 * gamma_function(g);
        let b = 255f32 * gamma_function(b);
        Rgb::new(r as u8, g as u8, b as u8)
    }
}
