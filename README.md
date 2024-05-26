# Rust utilities for color handling and conversion.

Allows conversion between

- Rgb/Rgba/Rgba1010102/Rgb565/RgbF16
- HSL
- HSV
- LAB
- LUV
- XYZ

### Performance

There are some prebuilt functions for ex.

```rust
srgb_to_lab(src_bytes, width * components, & mut xyz, width * 3 * std::mem::size_of::<f32>() as u32, width, height);
```

Prebuilt solutions ~3-5 times faster than naive implementation. If your case fits that you prebuilt function.
Speed increasing done with AVX, NEON and SSE, if you are disabled or not using CPU with this features then you won't
receive any benefits. 