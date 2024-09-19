/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[inline]
pub fn mlaf(x: f32, y: f32, z: f32) -> f32 {
    x.mul_add(y, z)
}
