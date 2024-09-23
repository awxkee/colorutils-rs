/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::euclidean::EuclideanDistance;
use crate::hsv::Hsv;
use crate::lab::Lab;
use crate::luv::Luv;
use crate::oklch::Oklch;
use crate::{
    adjust_saturation, clip_color, color_add, color_burn, color_darken, color_difference,
    color_dodge, color_exclusion, color_hard_light, color_hard_mix, color_lighten,
    color_linear_burn, color_linear_light, color_pin_light, color_reflect, color_screen,
    color_soft_light, color_soft_light_weight, color_vivid_light, pdf_lum, Hsl, Jzazbz, LAlphaBeta,
    LCh, Oklab, Rgba, Sigmoidal, TaxicabDistance, TransferFunction, Xyz,
};
use num_traits::{AsPrimitive, Bounded, Float, Num, Pow};
use std::cmp::{max, min, Ordering};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub};

#[repr(C)]
#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
/// Represents any RGB values, Rgb<u8>, Rgb<u16> etc.
pub struct Rgb<T> {
    /// Red component
    pub r: T,
    /// Green component
    pub g: T,
    /// Blue component
    pub b: T,
}

impl Rgb<u8> {
    /// Converts gamma corrected RGB to linear RGB
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert RGB into linear RGB
    #[inline]
    pub fn to_linear(&self, transfer_function: TransferFunction) -> Rgb<f32> {
        let linear_transfer = transfer_function.get_linearize_function();
        self.to_rgb_f32().apply(linear_transfer)
    }

    /// Converts gamma corrected RGB to linear RGB
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert RGB into linear RGB
    #[inline]
    pub fn from_linear(linear_rgb: Rgb<f32>, transfer_function: TransferFunction) -> Rgb<u8> {
        let linear_transfer = transfer_function.get_gamma_function();
        linear_rgb.apply(linear_transfer).to_u8()
    }

    /// Converts rgb to Jzazbz
    /// Here is luminance always considered 200 nits
    ///
    /// # Arguments
    /// `display_luminance` - display luminance
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn to_jzazbz(&self, transfer_function: TransferFunction) -> Jzazbz {
        Jzazbz::from_rgb(*self, transfer_function)
    }

    /// Converts rgb to Jzazbz
    ///
    /// # Arguments
    /// `display_luminance` - display luminance
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn to_jzazbz_with_luminance(
        &self,
        display_luminance: f32,
        transfer_function: TransferFunction,
    ) -> Jzazbz {
        Jzazbz::from_rgb_with_luminance(*self, display_luminance, transfer_function)
    }

    /// Converts rgb to XYZ
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert RGB into linear RGB
    #[inline]
    pub fn to_xyz(&self, matrix: &[[f32; 3]; 3], transfer_function: TransferFunction) -> Xyz {
        Xyz::from_rgb(*self, matrix, transfer_function)
    }

    /// Converts rgb to HSL
    #[inline]
    pub fn to_hsl(&self) -> Hsl {
        Hsl::from_rgb(*self)
    }

    /// Converts rgb to HSV
    #[inline]
    pub fn to_hsv(&self) -> Hsv {
        Hsv::from(*self)
    }

    /// Converts rgb to CIELAB
    #[inline]
    pub fn to_lab(&self) -> Lab {
        Lab::from_rgb(*self)
    }

    /// Converts rgb to CIELUV
    #[inline]
    pub fn to_luv(&self) -> Luv {
        Luv::from_rgb(*self)
    }

    /// Converts rgb to CIELCH
    #[inline]
    pub fn to_lch(&self) -> LCh {
        LCh::from_rgb(*self)
    }

    /// Converts rgb to RGB f32
    #[inline]
    pub fn to_rgb_f32(&self) -> Rgb<f32> {
        const SCALE: f32 = 1f32 / 255f32;
        Rgb::<f32>::new(
            self.r as f32 * SCALE,
            self.g as f32 * SCALE,
            self.b as f32 * SCALE,
        )
    }

    /// Converts rgb to *Oklab*
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn to_oklab(&self, transfer_function: TransferFunction) -> Oklab {
        Oklab::from_rgb(*self, transfer_function)
    }

    /// Converts rgb to *Oklch*
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn to_oklch(&self, transfer_function: TransferFunction) -> Oklch {
        Oklch::from_rgb(*self, transfer_function)
    }

    /// Converts rgb to S-shaped sigmoidized components
    #[inline]
    pub fn to_sigmoidal(&self) -> Sigmoidal {
        Sigmoidal::from_rgb(*self)
    }

    /// Converts rgb to *lαβ*
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function to convert into linear colorspace and backwards
    #[inline]
    pub fn to_lalphabeta(&self, transfer_function: TransferFunction) -> LAlphaBeta {
        LAlphaBeta::from_rgb(*self, transfer_function)
    }

    #[inline]
    pub fn blend_add(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_add(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_hsl_color(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_hsl_color(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_substract(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_substract(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_lighten(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_lighten(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_darken(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_darken(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_color_burn(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_color_burn(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_color_dodge(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_color_dodge(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_screen(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_screen(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_linear_light(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_linear_light(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_vivid_light(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_vivid_light(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_pin_light(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_pin_light(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_hard_mix(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_hard_mix(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_soft_light(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_soft_light(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn blend_exclusion(&self, other: Rgb<u8>) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let other_color = other.to_rgb_f32();
        let blended = in_color.blend_exclusion(other_color);
        blended.to_u8()
    }

    #[inline]
    pub fn saturation(&self, saturation: f32) -> Rgb<u8> {
        let in_color = self.to_rgb_f32();
        let saturated = in_color.saturation(saturation);
        saturated.to_u8()
    }
}

impl From<Rgb<f32>> for Rgb<u8> {
    #[inline]
    fn from(value: Rgb<f32>) -> Self {
        value.to_u8()
    }
}

impl Rgb<f32> {
    #[inline]
    pub fn apply(&self, gen: fn(f32) -> f32) -> Self {
        Self {
            r: gen(self.r),
            g: gen(self.g),
            b: gen(self.b),
        }
    }

    #[inline]
    pub fn to_u8(&self) -> Rgb<u8> {
        Rgb::<u8>::new(
            (self.r * 255f32).max(0f32).round().min(255f32) as u8,
            (self.g * 255f32).max(0f32).round().min(255f32) as u8,
            (self.b * 255f32).max(0f32).round().min(255f32) as u8,
        )
    }

    #[inline]
    pub fn clip_color(&self) -> Rgb<f32> {
        let (r, g, b) = clip_color!(self);
        Rgb::<f32>::new(r, g, b)
    }

    #[inline]
    fn pdf_set_lum(&self, new_lum: f32) -> Rgb<f32> {
        let d = new_lum - pdf_lum!(self);
        let r = self.r + d;
        let g = self.g + d;
        let b = self.b + d;
        let new_color = Rgb::<f32>::new(r, g, b);
        new_color.clip_color()
    }

    #[inline]
    pub fn blend_hsl_color(&self, backdrop: Rgb<f32>) -> Rgb<f32> {
        let lum = pdf_lum!(backdrop);
        self.pdf_set_lum(lum)
    }

    #[inline]
    /// The output of the ADD operator is the same for both bounded and unbounded source interpretations.
    /// Resulting color (xR) - (xaA + xaB)
    pub fn blend_add(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_add!(self.r, other.r);
        let new_g = color_add!(self.g, other.g);
        let new_b = color_add!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_substract(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_linear_burn!(self.r, other.r);
        let new_g = color_linear_burn!(self.g, other.g);
        let new_b = color_linear_burn!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_lighten(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_lighten!(self.r, other.r);
        let new_g = color_lighten!(self.g, other.g);
        let new_b = color_lighten!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_darken(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_darken!(self.r, other.r);
        let new_g = color_darken!(self.g, other.g);
        let new_b = color_darken!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_color_burn(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_burn!(self.r, other.r);
        let new_g = color_burn!(self.g, other.g);
        let new_b = color_burn!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_color_dodge(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_dodge!(self.r, other.r);
        let new_g = color_dodge!(self.g, other.g);
        let new_b = color_dodge!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_screen(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_screen!(self.r, other.r);
        let new_g = color_screen!(self.g, other.g);
        let new_b = color_screen!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_linear_light(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_linear_light!(self.r, other.r);
        let new_g = color_linear_light!(self.g, other.g);
        let new_b = color_linear_light!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_vivid_light(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_vivid_light!(self.r, other.r);
        let new_g = color_vivid_light!(self.g, other.g);
        let new_b = color_vivid_light!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_pin_light(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_pin_light!(self.r, other.r);
        let new_g = color_pin_light!(self.g, other.g);
        let new_b = color_pin_light!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_hard_mix(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_hard_mix!(self.r, other.r);
        let new_g = color_hard_mix!(self.g, other.g);
        let new_b = color_hard_mix!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_reflect(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_reflect!(self.r, other.r);
        let new_g = color_reflect!(self.g, other.g);
        let new_b = color_reflect!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_difference(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_difference!(self.r, other.r);
        let new_g = color_difference!(self.g, other.g);
        let new_b = color_difference!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_hard_light(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_hard_light!(self.r, other.r);
        let new_g = color_hard_light!(self.g, other.g);
        let new_b = color_hard_light!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_soft_light(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_soft_light!(self.r, other.r);
        let new_g = color_soft_light!(self.g, other.g);
        let new_b = color_soft_light!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn blend_exclusion(&self, other: Rgb<f32>) -> Rgb<f32> {
        let new_r = color_exclusion!(self.r, other.r);
        let new_g = color_exclusion!(self.g, other.g);
        let new_b = color_exclusion!(self.b, other.b);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn saturation(&self, saturation: f32) -> Rgb<f32> {
        let (new_r, new_g, new_b) = adjust_saturation!(self, saturation);
        Rgb::<f32>::new(new_r, new_g, new_b)
    }

    #[inline]
    pub fn grayscale(&self, grayscale_amount: f32) -> Rgb<f32> {
        let gray = self.r * 0.299f32 + self.g * 0.587 + self.b * 0.114;
        let new_r = self.r * (1.0 - grayscale_amount) + gray * grayscale_amount;
        let new_g = self.g * (1.0 - grayscale_amount) + gray * grayscale_amount;
        let new_b = self.b * (1.0 - grayscale_amount) + gray * grayscale_amount;
        Rgb::<f32>::new(new_r, new_g, new_b)
    }
}

impl From<Sigmoidal> for Rgb<u8> {
    #[inline]
    fn from(value: Sigmoidal) -> Self {
        value.to_rgb()
    }
}

impl<T> Rgb<T> {
    pub fn new(r: T, g: T, b: T) -> Rgb<T> {
        Rgb { r, g, b }
    }
}

impl<T> Rgb<T>
where
    T: Copy,
{
    pub fn dup(v: T) -> Rgb<T> {
        Rgb { r: v, g: v, b: v }
    }
}

impl<T> Index<usize> for Rgb<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            _ => panic!("Index out of bounds for Rgb"),
        }
    }
}

impl<T> IndexMut<usize> for Rgb<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            _ => panic!("Index out of bounds for RGB"),
        }
    }
}

macro_rules! generated_float_definition_rgb {
    ($T: ty) => {
        impl Rgb<$T> {
            #[inline]
            pub fn zeroed() -> Rgb<$T> {
                Rgb::<$T>::new(0., 0., 0.)
            }

            #[inline]
            pub fn ones() -> Rgb<$T> {
                Rgb::<$T>::new(1., 1., 1.)
            }

            #[inline]
            pub fn white() -> Rgb<$T> {
                Rgb::<$T>::ones()
            }

            #[inline]
            pub fn black() -> Rgb<$T> {
                Rgb::<$T>::zeroed()
            }

            #[inline]
            pub fn contrast(&self, contrast: $T) -> Rgb<$T> {
                let new_r = self.r * contrast + -0.5 * contrast + 0.5;
                let new_g = self.g * contrast + -0.5 * contrast + 0.5;
                let new_b = self.b * contrast + -0.5 * contrast + 0.5;
                Rgb::<$T>::new(new_r, new_g, new_b)
            }
        }
    };
}

generated_float_definition_rgb!(f32);
generated_float_definition_rgb!(f64);

macro_rules! generated_integral_definition_rgb {
    ($T: ty) => {
        impl Rgb<$T> {
            #[inline]
            pub fn zeroed() -> Rgb<$T> {
                Rgb::<$T>::new(0, 0, 0)
            }

            #[inline]
            pub fn capped() -> Rgb<$T> {
                Rgb::<$T>::new(<$T>::MAX, <$T>::MAX, <$T>::MAX)
            }

            #[inline]
            pub fn white() -> Rgb<$T> {
                Rgb::<$T>::capped()
            }

            #[inline]
            pub fn black() -> Rgb<$T> {
                Rgb::<$T>::new(0, 0, 0)
            }
        }
    };
}

generated_integral_definition_rgb!(u8);
generated_integral_definition_rgb!(u16);
generated_integral_definition_rgb!(i8);
generated_integral_definition_rgb!(i16);
generated_integral_definition_rgb!(i32);
generated_integral_definition_rgb!(u32);

macro_rules! generated_default_definition_rgb {
    ($T: ty) => {
        impl Default for Rgb<$T> {
            fn default() -> Self {
                Rgb::<$T>::zeroed()
            }
        }
    };
}

generated_default_definition_rgb!(u8);
generated_default_definition_rgb!(u16);
generated_default_definition_rgb!(i8);
generated_default_definition_rgb!(i16);
generated_default_definition_rgb!(i32);
generated_default_definition_rgb!(u32);
generated_default_definition_rgb!(f32);
generated_default_definition_rgb!(f64);

impl Rgb<u8> {
    #[inline]
    #[allow(clippy::manual_clamp)]
    pub fn contrast(&self, contrast: f32) -> Rgb<u8> {
        let new_r = (self.r as f32 * contrast + -127.5f32 * contrast + 127.5f32)
            .round()
            .min(255f32)
            .max(0f32);
        let new_g = (self.g as f32 * contrast + -127.5f32 * contrast + 127.5f32)
            .round()
            .min(255f32)
            .max(0f32);
        let new_b = (self.b as f32 * contrast + -127.5f32 * contrast + 127.5f32)
            .round()
            .min(255f32)
            .max(0f32);
        Rgb::<u8>::new(new_r as u8, new_g as u8, new_b as u8)
    }

    #[inline]
    pub fn grayscale(&self, grayscale_amount: f32) -> Rgb<u8> {
        let gray = self.r as f32 * 0.299f32 + self.g as f32 * 0.587 + self.b as f32 * 0.114;
        let new_r = self.r as f32 * (255f32 - grayscale_amount) + gray * grayscale_amount;
        let new_g = self.g as f32 * (255f32 - grayscale_amount) + gray * grayscale_amount;
        let new_b = self.b as f32 * (255f32 - grayscale_amount) + gray * grayscale_amount;
        Rgb::<u8>::new(
            new_r.round() as u8,
            new_g.round() as u8,
            new_b.round() as u8,
        )
    }
}

impl<T> Rgb<T>
where
    T: Copy,
{
    pub fn to_rgba(&self, a: T) -> Rgba<T> {
        Rgba::<T>::new(self.r, self.g, self.b, a)
    }
}

impl<T> EuclideanDistance for Rgb<T>
where
    T: Copy + AsPrimitive<f32>,
{
    fn euclidean_distance(&self, other: Rgb<T>) -> f32 {
        let dr = self.r.as_() - other.r.as_();
        let dg = self.g.as_() - other.g.as_();
        let db = self.b.as_() - other.b.as_();
        (dr * dr + dg * dg + db * db).sqrt()
    }
}

impl<T> TaxicabDistance for Rgb<T>
where
    T: Copy + AsPrimitive<f32>,
{
    fn taxicab_distance(&self, other: Self) -> f32 {
        let dr = self.r.as_() - other.r.as_();
        let dg = self.g.as_() - other.g.as_();
        let db = self.b.as_() - other.b.as_();
        dr.abs() + dg.abs() + db.abs()
    }
}

impl<T> Add for Rgb<T>
where
    T: Add<Output = T>,
{
    type Output = Rgb<T>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Rgb::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl<T> Sub for Rgb<T>
where
    T: Sub<Output = T>,
{
    type Output = Rgb<T>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Rgb::new(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}

impl<T> Div for Rgb<T>
where
    T: Div<Output = T>,
{
    type Output = Rgb<T>;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Rgb::new(self.r / rhs.r, self.g / rhs.g, self.b / rhs.b)
    }
}

impl<T> Mul for Rgb<T>
where
    T: Mul<Output = T>,
{
    type Output = Rgb<T>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Rgb::new(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b)
    }
}

impl<T> MulAssign for Rgb<T>
where
    T: MulAssign<T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.r *= rhs.r;
        self.g *= rhs.g;
        self.b *= rhs.b;
    }
}

macro_rules! generated_mul_assign_definition_rgb {
    ($T: ty) => {
        impl<T> MulAssign<$T> for Rgb<T>
        where
            T: MulAssign<$T>,
        {
            #[inline]
            fn mul_assign(&mut self, rhs: $T) {
                self.r *= rhs;
                self.g *= rhs;
                self.b *= rhs;
            }
        }
    };
}

generated_mul_assign_definition_rgb!(i8);
generated_mul_assign_definition_rgb!(u8);
generated_mul_assign_definition_rgb!(u16);
generated_mul_assign_definition_rgb!(i16);
generated_mul_assign_definition_rgb!(u32);
generated_mul_assign_definition_rgb!(i32);
generated_mul_assign_definition_rgb!(f32);
generated_mul_assign_definition_rgb!(f64);

impl<T> AddAssign for Rgb<T>
where
    T: AddAssign<T>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.r += rhs.r;
        self.g += rhs.g;
        self.b += rhs.b;
    }
}

macro_rules! generated_add_assign_definition_rgb {
    ($T: ty) => {
        impl<T> AddAssign<$T> for Rgb<T>
        where
            T: AddAssign<$T>,
        {
            #[inline]
            fn add_assign(&mut self, rhs: $T) {
                self.r += rhs;
                self.g += rhs;
                self.b += rhs;
            }
        }
    };
}

generated_add_assign_definition_rgb!(i8);
generated_add_assign_definition_rgb!(u8);
generated_add_assign_definition_rgb!(u16);
generated_add_assign_definition_rgb!(i16);
generated_add_assign_definition_rgb!(u32);
generated_add_assign_definition_rgb!(i32);
generated_add_assign_definition_rgb!(f32);
generated_add_assign_definition_rgb!(f64);

impl<T> DivAssign for Rgb<T>
where
    T: DivAssign<T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.r /= rhs.r;
        self.g /= rhs.g;
        self.b /= rhs.b;
    }
}

macro_rules! generated_div_assign_definition_rgb {
    ($T: ty) => {
        impl<T> DivAssign<$T> for Rgb<T>
        where
            T: DivAssign<$T>,
        {
            #[inline]
            fn div_assign(&mut self, rhs: $T) {
                self.r /= rhs;
                self.g /= rhs;
                self.b /= rhs;
            }
        }
    };
}

generated_div_assign_definition_rgb!(u8);
generated_div_assign_definition_rgb!(i8);
generated_div_assign_definition_rgb!(u16);
generated_div_assign_definition_rgb!(i16);
generated_div_assign_definition_rgb!(u32);
generated_div_assign_definition_rgb!(i32);
generated_div_assign_definition_rgb!(f32);
generated_div_assign_definition_rgb!(f64);

impl<T> Neg for Rgb<T>
where
    T: Neg<Output = T>,
{
    type Output = Rgb<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        Rgb::new(-self.r, -self.g, -self.b)
    }
}

impl<T> Rgb<T>
where
    T: Num + PartialOrd + Copy + Bounded + Ord,
{
    /// Clamp function to clamp each channel within a given range
    #[inline]
    #[allow(clippy::manual_clamp)]
    pub fn clamp(&self, min_value: T, max_value: T) -> Rgb<T> {
        Rgb::new(
            min(max(self.r, min_value), max_value),
            min(max(self.g, min_value), max_value),
            min(max(self.b, min_value), max_value),
        )
    }

    /// Min function to define min
    #[inline]
    pub fn min(&self, other_min: T) -> Rgb<T> {
        Rgb::new(
            min(self.r, other_min),
            min(self.g, other_min),
            min(self.b, other_min),
        )
    }

    /// Max function to define max
    #[inline]
    pub fn max(&self, other_max: T) -> Rgb<T> {
        Rgb::new(
            max(self.r, other_max),
            max(self.g, other_max),
            max(self.b, other_max),
        )
    }

    /// Clamp function to clamp each channel within a given range
    #[inline]
    #[allow(clippy::manual_clamp)]
    pub fn clamp_p(&self, min_value: Rgb<T>, max_value: Rgb<T>) -> Rgb<T> {
        Rgb::new(
            max(min(self.r, max_value.r), min_value.r),
            max(min(self.g, max_value.g), min_value.g),
            max(min(self.b, max_value.b), min_value.b),
        )
    }

    /// Min function to define min
    #[inline]
    pub fn min_p(&self, other_min: Rgb<T>) -> Rgb<T> {
        Rgb::new(
            min(self.r, other_min.r),
            min(self.g, other_min.g),
            min(self.b, other_min.b),
        )
    }

    /// Max function to define max
    #[inline]
    pub fn max_p(&self, other_max: Rgb<T>) -> Rgb<T> {
        Rgb::new(
            max(self.r, other_max.r),
            max(self.g, other_max.g),
            max(self.b, other_max.b),
        )
    }
}

impl<T> Rgb<T>
where
    T: Float + 'static,
    f32: AsPrimitive<T>,
{
    #[inline]
    pub fn sqrt(&self) -> Rgb<T> {
        let zeros = 0f32.as_();
        Rgb::new(
            if self.r.partial_cmp(&zeros).unwrap_or(Ordering::Less) == Ordering::Less {
                0f32.as_()
            } else {
                self.r.sqrt()
            },
            if self.g.partial_cmp(&zeros).unwrap_or(Ordering::Less) == Ordering::Less {
                0f32.as_()
            } else {
                self.g.sqrt()
            },
            if self.b.partial_cmp(&zeros).unwrap_or(Ordering::Less) == Ordering::Less {
                0f32.as_()
            } else {
                self.b.sqrt()
            },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Rgb<T> {
        Rgb::new(self.r.cbrt(), self.g.cbrt(), self.b.cbrt())
    }
}

impl<T> Pow<T> for Rgb<T>
where
    T: Float,
{
    type Output = Rgb<T>;

    #[inline]
    fn pow(self, rhs: T) -> Self::Output {
        Rgb::<T>::new(self.r.powf(rhs), self.g.powf(rhs), self.b.powf(rhs))
    }
}

impl<T> Pow<Rgb<T>> for Rgb<T>
where
    T: Float,
{
    type Output = Rgb<T>;

    #[inline]
    fn pow(self, rhs: Rgb<T>) -> Self::Output {
        Rgb::<T>::new(self.r.powf(rhs.r), self.g.powf(rhs.g), self.b.powf(rhs.b))
    }
}
