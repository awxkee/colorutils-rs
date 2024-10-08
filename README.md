# Rust utilities for color handling and conversion.

The goal is to provide support for common conversion and SIMD options for most common conversion path for high-performance

Available SIMD fast paths generally 5-10 times faster than naive implementations

Allows conversion between

- [x] Rgb/Rgba/Rgba1010102/Rgb565/RgbF16
- [x] HSL
- [x] HSV
- [x] CIE LAB
- [x] CIE LUV
- [x] CIE LCh
- [x] XYZ
- [x] Sigmoidal
- [x] Oklab
- [x] Oklch
- [x] Jzazbz
- [x] Jzczhz
- [x] lαβ (l-alpha-beta)

### Performance

There are some prebuilt functions for ex.

```rust
srgb_to_lab(src_bytes, width * components, &mut lab_store, width * 3 * std::mem::size_of::<f32>() as u32, width, height);
```

Prebuilt solutions ~3-5 times faster than naive implementation. If your case fits that you prebuilt function.
Speed increasing done with AVX, NEON and SSE, if you are disabled or not using CPU with this features then you won't
receive any benefits. 

Also, `fma` target feature for x86-64 is available.

Target feature at compile time `+avx2` must be activated to properly compile avx2 instructions. This is an important step even when runtime dispatch are used.

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
