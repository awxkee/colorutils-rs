/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use half::f16;

use crate::rgb::Rgb;
use crate::routines::{
    op_color_burn, op_color_dodge, op_darken, op_difference, op_exclusion, op_hard_light,
    op_hard_mix, op_lighten, op_linear_burn, op_linear_light, op_overlay, op_pin_light, op_reflect,
    op_screen, op_soft_light, op_vivid_light,
};
use crate::{adjust_saturation, clip_color, color_add, pdf_lum};

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
/// Represents any RGBA values, Rgba<u8>, Rgba<u16> etc.
pub struct Rgba<T> {
    /// Red component
    pub r: T,
    /// Green component
    pub g: T,
    /// Blue component
    pub b: T,
    /// Alpha component
    pub a: T,
}

impl Rgba<f16> {
    pub fn from_rgb(r: f16, g: f16, b: f16) -> Rgba<f16> {
        return Rgba {
            r,
            g,
            b,
            a: f16::from_f32(1f32),
        };
    }
}

impl<T> Rgba<T> {
    #[inline]
    pub fn new(r: T, g: T, b: T, a: T) -> Rgba<T> {
        return Rgba { r, g, b, a };
    }
}

impl Rgba<f32> {
    #[inline]
    pub fn zeroed() -> Rgba<f32> {
        Rgba::<f32>::new(0., 0., 0., 0.)
    }

    #[inline]
    pub fn ones() -> Rgba<f32> {
        Rgba::<f32>::new(1., 1., 1., 1.)
    }

    #[inline]
    pub fn white() -> Rgba<f32> {
        Rgba::<f32>::new(1., 1., 1., 1.)
    }

    #[inline]
    pub fn black() -> Rgba<f32> {
        Rgba::<f32>::new(0., 0., 0., 1.)
    }

    #[inline]
    pub fn from_rgb(r: f32, g: f32, b: f32) -> Rgba<f32> {
        return Rgba { r, g, b, a: 1f32 };
    }
}

impl Rgba<u8> {
    #[inline]
    pub fn zeroed() -> Rgba<u8> {
        Rgba::<u8>::new(0, 0, 0, 0)
    }

    #[inline]
    pub fn capped() -> Rgba<u8> {
        Rgba::<u8>::new(255, 255, 255, 255)
    }

    #[inline]
    pub fn black() -> Rgba<u8> {
        Rgba::<u8>::new(0, 0, 0, 255)
    }

    #[inline]
    pub fn white() -> Rgba<u8> {
        Rgba::<u8>::capped()
    }
}

impl Rgba<f32> {
    /// Using alpha blend over algorithm where current color is on bottom ( destination )
    #[inline]
    pub fn blend_over_alpha(&self, color_foreground: Rgba<f32>) -> Rgba<f32> {
        let a_dst = self.a + color_foreground.a * (1. - self.a);
        if a_dst == 0. {
            return Rgba::<f32>::zeroed();
        }
        let a_dst_recpeq = 1. / a_dst;
        let out_r = ((1. - self.a) * color_foreground.a * color_foreground.r + self.a * self.r)
            * a_dst_recpeq;
        let out_g = ((1. - self.a) * color_foreground.a * color_foreground.g + self.a * self.g)
            * a_dst_recpeq;
        let out_b = ((1. - self.a) * color_foreground.a * color_foreground.b + self.a * self.b)
            * a_dst_recpeq;
        Rgba::<f32>::new(out_r, out_g, out_b, a_dst)
    }

    /// Using alpha blend over algorithm where current color is on bottom ( destination )
    /// aR = aA + aB·(1−aA)
    /// xR = 1/aR · [ (1−aB)·xaA + (1−aA)·xaB + aA·aB·f(xA,xB) ]
    #[inline]
    pub fn blend_over_with_op(
        &self,
        color_foreground: Rgba<f32>,
        op: fn(f32, f32) -> f32,
    ) -> Rgba<f32> {
        let a_dst = self.a + color_foreground.a * (1. - self.a);
        if a_dst == 0. {
            return Rgba::<f32>::zeroed();
        }
        let a_dst_recpeq = 1. / a_dst;
        let new_r = if a_dst != 0. {
            (1. - color_foreground.a) * self.r
                + (1. - self.a) * color_foreground.r
                + self.a * color_foreground.a * op(self.r, color_foreground.r)
        } else {
            0.
        } * a_dst_recpeq;
        let new_g = if a_dst != 0. {
            (1. - color_foreground.a) * self.g
                + (1. - self.a) * color_foreground.g
                + self.a * color_foreground.a * op(self.g, color_foreground.g)
        } else {
            0.
        } * a_dst_recpeq;
        let new_b = if a_dst != 0. {
            (1. - color_foreground.a) * self.b
                + (1. - self.a) * color_foreground.b
                + self.a * color_foreground.a * op(self.b, color_foreground.b)
        } else {
            0.
        } * a_dst_recpeq;

        Rgba::<f32>::new(new_r, new_g, new_b, a_dst)
    }

    #[inline]
    pub fn blend_overlay(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_overlay)
    }

    #[inline]
    /// The output of the ADD operator is the same for both bounded and unbounded source interpretations.
    /// Resulting alpha (aR) - min(1, aA+aB)
    /// Resulting color (xR) - (xaA + xaB)/aR
    pub fn blend_add(&self, other: Rgba<f32>) -> Rgba<f32> {
        let new_r = color_add!(self.r, other.r);
        let new_g = color_add!(self.g, other.g);
        let new_b = color_add!(self.b, other.b);
        let alpha = 1.0f32.min(self.a + other.a);
        if alpha == 0. {
            return Rgba::<f32>::zeroed();
        }
        let recprec_alpha = 1. / alpha;
        Rgba::<f32>::new(
            new_r * recprec_alpha,
            new_g * recprec_alpha,
            new_b * recprec_alpha,
            alpha,
        )
    }

    #[inline]
    /// The output of the ADD operator is the same for both bounded and unbounded source interpretations.
    /// Resulting alpha (aR) - min(1, aA+aB)
    /// Resulting color (xR) - (min(aA, 1−aB)·xA + xaB)/aR
    pub fn blend_saturate(&self, other: Rgba<f32>) -> Rgba<f32> {
        let alpha = 1.0f32.min(self.a + other.a);
        let src = self;
        let dst = other;
        let recip_alpha = 1. / alpha;
        let new_r = if alpha != 0.0 {
            (src.r * src.a.min(1.0 - dst.a) + dst.r * dst.a) * recip_alpha
        } else {
            0.0
        };
        let new_g = if alpha != 0.0 {
            (src.g * src.a.min(1.0 - dst.a) + dst.g * dst.a) * recip_alpha
        } else {
            0.0
        };
        let new_b = if alpha != 0.0 {
            (src.b * src.a.min(1.0 - dst.a) + dst.b * dst.a) * recip_alpha
        } else {
            0.0
        };
        Rgba::<f32>::new(new_r, new_g, new_b, alpha)
    }

    #[inline]
    pub fn blend_substract(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_linear_burn)
    }

    #[inline]
    pub fn blend_lighten(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_lighten)
    }

    #[inline]
    pub fn blend_darken(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_darken)
    }

    #[inline]
    pub fn blend_color_burn(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_color_burn)
    }

    #[inline]
    pub fn blend_color_dodge(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_color_dodge)
    }

    #[inline]
    pub fn blend_screen(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_screen)
    }

    #[inline]
    pub fn blend_linear_light(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_linear_light)
    }

    #[inline]
    pub fn blend_vivid_light(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_vivid_light)
    }

    #[inline]
    pub fn blend_pin_light(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_pin_light)
    }

    #[inline]
    pub fn blend_hard_mix(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_hard_mix)
    }

    #[inline]
    pub fn blend_reflect(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_reflect)
    }

    #[inline]
    pub fn blend_difference(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_difference)
    }

    #[inline]
    pub fn blend_hard_light(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_hard_light)
    }

    #[inline]
    pub fn blend_soft_light(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_soft_light)
    }

    #[inline]
    pub fn blend_exclusion(&self, other: Rgba<f32>) -> Rgba<f32> {
        self.blend_over_with_op(other, op_exclusion)
    }

    #[inline]
    pub fn blend_in(&self, other: Rgba<f32>) -> Rgba<f32> {
        Rgba::<f32>::new(self.r, self.g, self.b, other.a * self.a)
    }

    #[inline]
    pub fn blend_out(&self, other: Rgba<f32>) -> Rgba<f32> {
        Rgba::<f32>::new(self.r, self.g, self.b, (1. - other.a) * self.a)
    }

    #[inline]
    pub fn blend_atop(&self, other: Rgba<f32>) -> Rgba<f32> {
        let new_r = self.r + other.r * (1. - self.a);
        let new_g = self.g + other.g * (1. - self.a);
        let new_b = self.b + other.b * (1. - self.a);
        Rgba::<f32>::new(new_r, new_g, new_b, other.a)
    }

    #[inline]
    pub fn blend_dest_out(&self, other: Rgba<f32>) -> Rgba<f32> {
        Rgba::<f32>::new(other.r, other.g, other.b, (1. - self.a) * other.a)
    }

    #[inline]
    pub fn blend_dest_atop(&self, other: Rgba<f32>) -> Rgba<f32> {
        let new_r = other.r + self.r * (1. - other.a);
        let new_g = other.g + self.g * (1. - other.a);
        let new_b = other.b + self.b * (1. - other.a);
        Rgba::<f32>::new(new_r, new_g, new_b, self.a)
    }

    #[inline]
    /// The output of the ADD operator is the same for both bounded and unbounded source interpretations.
    /// Resulting alpha (aR) - aA + aB − 2·aA·aB
    /// Resulting color (xR) - (xaA·(1−aB) + xaB·(1−aA))/aR
    pub fn blend_xor(&self, other: Rgba<f32>) -> Rgba<f32> {
        let alpha = self.a + other.a - 2. * self.a * other.a;
        let recpec_alpha = 1. / alpha;
        let new_r = if alpha != 0.0 {
            (self.r * (1. - other.a) + other.r * (1. - self.a)) * recpec_alpha
        } else {
            0.0
        };
        let new_g = if alpha != 0.0 {
            (self.g * (1. - other.a) + other.g * (1. - self.a)) * recpec_alpha
        } else {
            0.0
        };
        let new_b = if alpha != 0.0 {
            (self.b * (1. - other.a) + other.b * (1. - self.a)) * recpec_alpha
        } else {
            0.0
        };
        Rgba::<f32>::new(new_r, new_g, new_b, alpha)
    }

    #[inline]
    pub fn clip_color(&self) -> Rgba<f32> {
        let (r, g, b) = clip_color!(self);
        return Rgba::<f32>::new(r, g, b, self.a);
    }

    #[inline]
    fn pdf_set_lum(&self, new_lum: f32) -> Rgba<f32> {
        let d = new_lum - pdf_lum!(self);
        let r = self.r + d;
        let g = self.g + d;
        let b = self.b + d;
        let new_color = Rgba::<f32>::new(r, g, b, self.a);
        return new_color.clip_color();
    }

    #[inline]
    fn pdf_sat(&self) -> f32 {
        self.r.max(self.g).max(self.b) - self.r.min(self.g).min(self.b)
    }

    #[inline]
    fn pdf_set_sat(&self, s: f32) -> Rgba<f32> {
        let mut cmax = self.r.max(self.g).max(self.b);
        let cmin = self.r.min(self.g).min(self.b);

        let mut cmid = if self.r != cmax && self.r != cmin {
            self.r
        } else if self.g != cmax && self.g != cmin {
            self.g
        } else {
            self.b
        };

        if cmax > cmin {
            cmid = ((cmid - cmin) * s) / (cmax - cmin);
            cmax = s;
        } else {
            cmid = 0.;
            cmax = 0.;
        }

        // Cmin is always set to 0
        let cmin = 0.0;

        // Construct the new color
        let r = if cmax > cmin {
            cmin + (self.r - cmin) * cmid
        } else {
            cmin
        };
        let g = if cmax > cmin {
            cmin + (self.g - cmin) * cmid
        } else {
            cmin
        };
        let b = if cmax > cmin {
            cmin + (self.b - cmin) * cmid
        } else {
            cmin
        };

        return Rgba::<f32>::new(r, g, b, self.a);
    }

    #[inline]
    pub fn blend_hsl_color(&self, backdrop: Rgba<f32>) -> Rgba<f32> {
        let lum = pdf_lum!(backdrop);
        self.pdf_set_lum(lum)
    }

    #[inline]
    pub fn blend_hsl_saturation(&self, backdrop: Rgba<f32>) -> Rgba<f32> {
        let j1 = backdrop.pdf_set_sat(self.pdf_sat());
        j1.pdf_set_lum(pdf_lum!(backdrop))
    }

    #[inline]
    pub fn contrast(&self, contrast: f32) -> Rgba<f32> {
        let new_r = self.r * contrast + -0.5f32 * contrast + 0.5f32;
        let new_g = self.g * contrast + -0.5f32 * contrast + 0.5f32;
        let new_b = self.b * contrast + -0.5f32 * contrast + 0.5f32;
        Rgba::<f32>::new(new_r, new_g, new_b, self.a)
    }

    #[inline]
    pub fn saturation(&self, saturation: f32) -> Rgba<f32> {
        let (new_r, new_g, new_b) = adjust_saturation!(self, saturation);
        Rgba::<f32>::new(new_r, new_g, new_b, self.a)
    }

    #[inline]
    pub fn grayscale(&self, grayscale_amount: f32) -> Rgba<f32> {
        let gray = self.r * 0.299f32 + self.g * 0.587 + self.b * 0.114;
        let new_r = self.r * (1.0 - grayscale_amount) + gray * grayscale_amount;
        let new_g = self.g * (1.0 - grayscale_amount) + gray * grayscale_amount;
        let new_b = self.b * (1.0 - grayscale_amount) + gray * grayscale_amount;
        Rgba::<f32>::new(new_r, new_g, new_b, self.a)
    }
}

impl Rgba<u8> {
    #[inline]
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Rgba<u8> {
        return Rgba {
            r,
            g,
            b,
            a: u8::MAX,
        };
    }

    #[inline]
    pub fn to_rgb(&self) -> Rgb<u8> {
        Rgb {
            r: self.r,
            g: self.g,
            b: self.b,
        }
    }

    /// Using alpha blend over algorithm where current color is on bottom ( destination )
    #[inline]
    pub fn blend_over_alpha(&self, color_foreground: Rgba<u8>) -> Rgba<u8> {
        let destination_f = self.to_rgba_f32();
        let source_f = color_foreground.to_rgba_f32();
        let blended = destination_f.blend_over_alpha(source_f);
        blended.to_rgba8()
    }

    /// f(cA,cB) = set_lum(cA, lum(cB))
    #[inline]
    pub fn blend_hsl_color(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = source.blend_hsl_color(backdrop);
        blended.to_rgba8()
    }

    /// f(cA,cB) = set_lum(cB, lum(cA))
    #[inline]
    pub fn blend_hsl_lumonosity(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_hsl_color(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_overlay(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_overlay(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_add(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_add(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_substract(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_substract(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_lighten(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_lighten(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_darken(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_darken(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_color_burn(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_color_burn(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_color_dodge(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_color_dodge(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_screen(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_screen(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_linear_light(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_linear_light(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_vivid_light(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_vivid_light(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_pin_light(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_pin_light(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_hard_mix(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_hard_mix(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_reflect(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_reflect(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_difference(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_difference(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_saturate(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_saturate(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_hard_light(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_hard_light(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_soft_light(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_soft_light(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_exclusion(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_exclusion(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_in(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_in(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_out(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_out(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_atop(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_atop(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_dest_out(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_dest_out(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_dest_atop(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_dest_atop(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn blend_xor(&self, backdrop: Rgba<u8>) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let backdrop = backdrop.to_rgba_f32();
        let blended = backdrop.blend_xor(source);
        blended.to_rgba8()
    }

    #[inline]
    pub fn contrast(&self, contrast: f32) -> Rgba<u8> {
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
        Rgba::<u8>::new(new_r as u8, new_g as u8, new_b as u8, self.a)
    }

    #[inline]
    pub fn saturation(&self, saturation: f32) -> Rgba<u8> {
        let source = self.to_rgba_f32();
        let saturated = source.saturation(saturation);
        saturated.to_rgba8()
    }

    #[inline]
    pub fn grayscale(&self, grayscale_amount: f32) -> Rgba<u8> {
        let gray = self.r as f32 * 0.299f32 + self.g as f32 * 0.587 + self.b as f32 * 0.114;
        let new_r = self.r as f32 * (255f32 - grayscale_amount) + gray * grayscale_amount;
        let new_g = self.g as f32 * (255f32 - grayscale_amount) + gray * grayscale_amount;
        let new_b = self.b as f32 * (255f32 - grayscale_amount) + gray * grayscale_amount;
        Rgba::<u8>::new(
            new_r.round() as u8,
            new_g.round() as u8,
            new_b.round() as u8,
            self.a,
        )
    }
}

pub trait ToRgba8 {
    fn to_rgba8(&self) -> Rgba<u8>;
}

pub trait ToRgbaF16 {
    fn to_rgba_f16(&self) -> Rgba<f16>;
}

pub trait ToRgb565 {
    fn to_rgb_565(&self) -> Rgb565;
}

pub trait ToRgbaF32 {
    fn to_rgba_f32(&self) -> Rgba<f32>;
}

impl ToRgbaF32 for Rgba<u8> {
    #[inline]
    fn to_rgba_f32(&self) -> Rgba<f32> {
        const SCALE_U8: f32 = 1f32 / 255f32;
        return Rgba::<f32>::new(
            self.r as f32 * SCALE_U8,
            self.g as f32 * SCALE_U8,
            self.b as f32 * SCALE_U8,
            self.a as f32 * SCALE_U8,
        );
    }
}

impl ToRgba8 for Rgba<f32> {
    #[inline]
    fn to_rgba8(&self) -> Rgba<u8> {
        return Rgba {
            r: (self.r * 255f32).min(255f32).max(0f32) as u8,
            g: (self.g * 255f32).min(255f32).max(0f32) as u8,
            b: (self.b * 255f32).min(255f32).max(0f32) as u8,
            a: (self.a * 255f32).min(255f32).max(0f32) as u8,
        };
    }
}

impl ToRgba8 for Rgba<f16> {
    #[inline]
    fn to_rgba8(&self) -> Rgba<u8> {
        return Rgba {
            r: (self.r.to_f32() * 255f32).min(255f32).max(0f32) as u8,
            g: (self.g.to_f32() * 255f32).min(255f32).max(0f32) as u8,
            b: (self.b.to_f32() * 255f32).min(255f32).max(0f32) as u8,
            a: (self.a.to_f32() * 255f32).min(255f32).max(0f32) as u8,
        };
    }
}

impl ToRgbaF16 for Rgba<f32> {
    #[inline]
    fn to_rgba_f16(&self) -> Rgba<f16> {
        Rgba {
            r: f16::from_f32(self.r),
            g: f16::from_f32(self.g),
            b: f16::from_f32(self.b),
            a: f16::from_f32(self.a),
        }
    }
}

static SCALE_U8_F32: f32 = 1f32 / 255f32;

impl ToRgbaF16 for Rgba<u8> {
    #[inline]
    fn to_rgba_f16(&self) -> Rgba<f16> {
        Rgba {
            r: f16::from_f32(self.r as f32 * SCALE_U8_F32),
            g: f16::from_f32(self.g as f32 * SCALE_U8_F32),
            b: f16::from_f32(self.b as f32 * SCALE_U8_F32),
            a: f16::from_f32(self.a as f32 * SCALE_U8_F32),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
/// Represents RGB 565 color in one u16
pub struct Rgb565 {
    pub rgb565: u16,
}

impl Rgb565 {
    pub fn new(color: u16) -> Rgb565 {
        Rgb565 { rgb565: color }
    }
}

impl ToRgba8 for Rgb565 {
    #[inline]
    fn to_rgba8(&self) -> Rgba<u8> {
        let red8 = ((self.rgb565 & 0b1111100000000000) >> 8) as u8;
        let green8 = ((self.rgb565 & 0b11111100000) >> 3) as u8;
        let blue8 = ((self.rgb565 & 0b11111) << 3) as u8;
        Rgba::<u8>::new(red8, green8, blue8, u8::MAX)
    }
}

static SCALE_RGB565_5BIT: f32 = 1f32 / 31f32;
static SCALE_RGB565_6BIT: f32 = 1f32 / 63f32;

impl ToRgbaF16 for Rgb565 {
    #[inline]
    fn to_rgba_f16(&self) -> Rgba<f16> {
        let red5 = (self.rgb565 & 0b1111100000000000) as f32 * SCALE_RGB565_5BIT;
        let green6 = (self.rgb565 & 0b11111100000) as f32 * SCALE_RGB565_6BIT;
        let blue5 = (self.rgb565 & 0b11111) as f32 * SCALE_RGB565_5BIT;
        Rgba::<f16>::from_rgb(
            f16::from_f32(red5),
            f16::from_f32(green6),
            f16::from_f32(blue5),
        )
    }
}

impl ToRgb565 for Rgba<u8> {
    #[inline]
    fn to_rgb_565(&self) -> Rgb565 {
        let red565 = ((self.r as u16) >> 3) << 11;
        let green565 = ((self.g as u16) >> 2) << 5;
        let blue565 = (self.b as u16) >> 3;
        Rgb565 {
            rgb565: red565 | green565 | blue565,
        }
    }
}

impl ToRgb565 for Rgba<f16> {
    #[inline]
    fn to_rgb_565(&self) -> Rgb565 {
        let red5 = (self.r.to_f32() * 31f32).max(31f32).min(0f32) as u16;
        let green6 = (self.g.to_f32() * 63f32).max(63f32).min(0f32) as u16;
        let blue5 = (self.b.to_f32() * 31f32).max(31f32).min(0f32) as u16;
        Rgb565 {
            rgb565: red5 | green6 | blue5,
        }
    }
}

impl ToRgb565 for Rgba<f32> {
    #[inline]
    fn to_rgb_565(&self) -> Rgb565 {
        let red5 = (self.r * 31f32).max(31f32).min(0f32) as u16;
        let green6 = (self.g * 63f32).max(63f32).min(0f32) as u16;
        let blue5 = (self.b * 31f32).max(31f32).min(0f32) as u16;
        Rgb565 {
            rgb565: red5 | green6 | blue5,
        }
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Represents RGBA 1010102 in one u32 store
pub struct Rgba1010102 {
    pub rgba: u32,
}

impl Rgba1010102 {
    #[inline]
    pub fn new(color: u32) -> Rgba1010102 {
        Rgba1010102 { rgba: color }
    }
}

impl ToRgba8 for Rgba1010102 {
    #[inline]
    fn to_rgba8(&self) -> Rgba<u8> {
        let mask = (1u32 << 10u32) - 1u32;
        let r = (self.rgba) & mask;
        let g = (self.rgba >> 10) & mask;
        let b = (self.rgba >> 20) & mask;
        let a = (self.rgba >> 30) & 0b00000011;
        Rgba::<u8>::new(
            (r >> 2) as u8,
            (g >> 2) as u8,
            (b >> 2) as u8,
            (a << 6) as u8,
        )
    }
}

static SCALE_RGBA10: f32 = 1f32 / 1023f32;
static SCALE_RGBA10ALPHA: f32 = 1f32 / 3f32;

impl ToRgbaF16 for Rgba1010102 {
    #[inline]
    fn to_rgba_f16(&self) -> Rgba<f16> {
        let mask = (1u32 << 10u32) - 1u32;
        let r = (self.rgba) & mask;
        let g = (self.rgba >> 10) & mask;
        let b = (self.rgba >> 20) & mask;
        let a = (self.rgba >> 30) & 0b00000011;
        Rgba::<f16>::new(
            f16::from_f32(r as f32 * SCALE_RGBA10),
            f16::from_f32(g as f32 * SCALE_RGBA10),
            f16::from_f32(b as f32 * SCALE_RGBA10),
            f16::from_f32(a as f32 * SCALE_RGBA10ALPHA),
        )
    }
}

pub trait ToRgba1010102 {
    #[allow(dead_code)]
    fn to_rgba1010102(&self) -> Rgba1010102;
}

impl ToRgba1010102 for Rgba<u8> {
    #[inline]
    fn to_rgba1010102(&self) -> Rgba1010102 {
        let r = (self.r as u32) << 2;
        let g = (self.g as u32) << 2;
        let b = (self.b as u32) << 2;
        let a = (self.a as u32) >> 6;
        let rgba1010102 = (a << 30) | (r << 20) | (g << 10) | b;
        Rgba1010102 { rgba: rgba1010102 }
    }
}

impl ToRgba1010102 for Rgba<f16> {
    #[inline]
    fn to_rgba1010102(&self) -> Rgba1010102 {
        let r = (self.r.to_f32() * 1023f32).min(1023f32).max(0f32) as u32;
        let g = (self.g.to_f32() * 1023f32).min(1023f32).max(0f32) as u32;
        let b = (self.b.to_f32() * 1023f32).min(1023f32).max(0f32) as u32;
        let a = (self.a.to_f32() * 3f32).min(3f32).max(0f32) as u32;
        let rgba1010102 = (a << 30) | (r << 20) | (g << 10) | b;
        Rgba1010102 { rgba: rgba1010102 }
    }
}
