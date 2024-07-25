/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[macro_export]
macro_rules! color_dodge {
    ($base: expr, $other: expr) => {{
        if $other == 1.0 {
            $other
        } else {
            ($base / (1.0 - $other)).min(1.)
        }
    }};
}

#[macro_export]
macro_rules! color_linear_burn {
    ($base: expr, $other: expr) => {{
        ($base + $other - 1.).max(0.)
    }};
}

#[macro_export]
macro_rules! color_burn {
    ($base: expr, $other: expr) => {{
        if $other == 0.0 {
            $other
        } else {
            (1.0 - ((1.0 - $base) / $other)).max(0.)
        }
    }};
}

#[macro_export]
macro_rules! color_darken {
    ($base: expr, $other: expr) => {{
        $base.min($other)
    }};
}

#[macro_export]
macro_rules! color_lighten {
    ($base: expr, $other: expr) => {{
        $base.max($other)
    }};
}

#[macro_export]
macro_rules! color_screen {
    ($base: expr, $other: expr) => {{
        $base + $other - $base * $other
    }};
}

#[macro_export]
macro_rules! color_add {
    ($base: expr, $other: expr) => {{
        ($base + $other).min(1.)
    }};
}

#[macro_export]
macro_rules! color_linear_light {
    ($base: expr, $other: expr) => {{
        if $other < 0.5 {
            color_linear_burn!($base, (2.0 * $other))
        } else {
            color_add!($base, (2.0 * ($other - 0.5)))
        }
    }};
}

#[macro_export]
macro_rules! color_vivid_light {
    ($base: expr, $other: expr) => {{
        if $other < 0.5 {
            color_burn!($base, (2.0 * $other))
        } else {
            color_dodge!($base, (2.0 * ($other - 0.5)))
        }
    }};
}

#[macro_export]
macro_rules! color_pin_light {
    ($base: expr, $other: expr) => {{
        if $other < 0.5 {
            color_darken!($base, (2.0 * $other))
        } else {
            color_lighten!($base, (2.0 * ($other - 0.5)))
        }
    }};
}

#[macro_export]
macro_rules! color_hard_mix {
    ($base: expr, $other: expr) => {{
        if color_vivid_light!($base, $other) < 0.5 {
            0.
        } else {
            1.
        }
    }};
}

#[macro_export]
macro_rules! color_reflect {
    ($base: expr, $other: expr) => {{
        if $other == 1.0 {
            $other
        } else {
            ($base * $base / (1.0 - $other)).min(1.)
        }
    }};
}

#[macro_export]
macro_rules! color_difference {
    ($base: expr, $other: expr) => {{
        ($base - $other).abs()
    }};
}

#[macro_export]
macro_rules! pdf_lum {
    ($base: expr) => {{
        0.3 * $base.r + 0.59 * $base.g + 0.11 * $base.g
    }};
}

#[macro_export]
macro_rules! clip_color {
    ($base: expr) => {{
        let l = pdf_lum!($base);
        let n = $base.r.min($base.g).min($base.b);
        let x = $base.r.max($base.g).max($base.b);
        let (mut r, mut g, mut b) = ($base.r, $base.g, $base.b);
        if n < 0.0 {
            let recip = 1. / (l - n);
            r = l + (((r - l) * l) * recip);
            g = l + (((g - l) * l) * recip);
            b = l + (((b - l) * l) * recip);
        }
        if x > 1.0 {
            let recip = 1. / (x - l);
            r = l + (((r - l) * (1. - l)) * recip);
            g = l + (((g - l) * (1. - l)) * recip);
            b = l + (((b - l) * (1. - l)) * recip);
        }
        (r, g, b)
    }};
}

#[macro_export]
macro_rules! color_hard_light {
    ($base: expr, $other: expr) => {{
        if $base <= 0.5 {
            2. * $base * $other
        } else {
            1. - 2. * (1. - $base) * (1. - $other)
        }
    }};
}

#[macro_export]
macro_rules! color_soft_light_weight {
    ($x: expr) => {{
        if ($x <= 0.25) {
            ((16. * $x - 12.) * $x + 4.) * $x
        } else {
            $x.sqrt()
        }
    }};
}

#[macro_export]
macro_rules! color_soft_light {
    ($base: expr, $other: expr) => {{
        if ($base <= 0.5) {
            $other - (1. - 2. * $base) * $other * (1. - $other)
        } else {
            $other + (2.0 * $base - 1.) * (color_soft_light_weight!($other) - $other)
        }
    }};
}

#[macro_export]
macro_rules! color_exclusion {
    ($base: expr, $other: expr) => {{
        $base + $other - 2. * $base * $other
    }};
}

#[macro_export]
macro_rules! adjust_saturation {
    ($store: expr, $saturation: expr) => {{
        let c1 = 0.213 + 0.787 * $saturation;
        let c2 = 0.715 - 0.715 * $saturation;
        let c3 = 0.072 - 0.072 * $saturation;

        let c4 = 0.213 - 0.213 * $saturation;
        let c5 = 0.715 + 0.285 * $saturation;
        let c6 = 0.072 - 0.072 * $saturation;

        let c7 = 0.213 - 0.213 * $saturation;
        let c8 = 0.715 - 0.715 * $saturation;
        let c9 = 0.072 + 0.928 * $saturation;
        let r1 = $store.r * c1 + $store.g * c2 + $store.b * c3;
        let g1 = $store.r * c4 + $store.g * c5 + $store.b * c6;
        let b1 = $store.r * c7 + $store.g * c8 + $store.b * c9;
        (r1, g1, b1)
    }};
}

pub(crate) fn op_color_dodge(a: f32, b: f32) -> f32 {
    if b == 1.0 {
        b
    } else {
        (a / (1.0 - b)).min(1.)
    }
}

pub(crate) fn op_screen(a: f32, b: f32) -> f32 {
    color_screen!(a, b)
}

pub(crate) fn op_color_burn(a: f32, b: f32) -> f32 {
    color_burn!(a, b)
}

pub(crate) fn op_darken(a: f32, b: f32) -> f32 {
    a.min(b)
}

pub(crate) fn op_lighten(a: f32, b: f32) -> f32 {
    a.max(b)
}

pub(crate) fn op_linear_burn(a: f32, b: f32) -> f32 {
    color_linear_burn!(a, b)
}

pub(crate) fn op_reflect(a: f32, b: f32) -> f32 {
    color_reflect!(a, b)
}

pub(crate) fn op_overlay(a: f32, b: f32) -> f32 {
    if b < 0.5 {
        2.0 * a * b
    } else {
        1.0 - 2.0 * (1.0 - a) * (1.0 - b)
    }
}

pub(crate) fn op_difference(a: f32, b: f32) -> f32 {
    color_difference!(a, b)
}

pub(crate) fn op_exclusion(a: f32, b: f32) -> f32 {
    color_exclusion!(a, b)
}

pub(crate) fn op_linear_light(a: f32, b: f32) -> f32 {
    color_linear_light!(a, b)
}

pub(crate) fn op_vivid_light(a: f32, b: f32) -> f32 {
    color_vivid_light!(a, b)
}

pub(crate) fn op_pin_light(a: f32, b: f32) -> f32 {
    color_pin_light!(a, b)
}

pub(crate) fn op_hard_mix(a: f32, b: f32) -> f32 {
    color_hard_mix!(a, b)
}

pub(crate) fn op_hard_light(a: f32, b: f32) -> f32 {
    color_hard_light!(a, b)
}

pub(crate) fn op_soft_light(a: f32, b: f32) -> f32 {
    color_soft_light!(a, b)
}
