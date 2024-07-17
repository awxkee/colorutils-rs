/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

/// Computes taxicab distance for color, better works for L*a*b
pub trait TaxicabDistance {
    fn taxicab_distance(&self, other: Self) -> f32;
}
