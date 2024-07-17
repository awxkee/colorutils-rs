/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

/// Trait that implements Euclidean distance for color
pub trait EuclideanDistance {
    fn euclidean_distance(&self, other: Self) -> f32;
}
