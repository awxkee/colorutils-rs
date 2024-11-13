/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use num_traits::MulAdd;
use std::ops::{Add, Mul};

#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
))]
#[inline(always)]
pub fn mlaf<T: Copy + Mul<T, Output = T> + Add<T, Output = T> + MulAdd<T, Output = T>>(
    acc: T,
    a: T,
    b: T,
) -> T {
    MulAdd::mul_add(a, b, acc)
}

#[inline(always)]
#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
pub fn mlaf<T: Copy + Mul<T, Output = T> + Add<T, Output = T> + MulAdd<T, Output = T>>(
    acc: T,
    a: T,
    b: T,
) -> T {
    acc + a * b
}
