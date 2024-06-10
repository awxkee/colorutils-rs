use std::arch::aarch64::{vdupq_n_f32, vdupq_n_u32, vgetq_lane_f32, vgetq_lane_u32};
use colorutils_rs::*;
use image::io::Reader as ImageReader;
use image::{EncodableLayout, GenericImageView};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

fn main() {
    // #[cfg(target_arch = "x86_64")]
    // unsafe {
    //     println!("HAS fma: {}", is_x86_feature_detected!("fma"));
    //     let mut dst: [f32; 4] = [0f32; 4];
    //     // let src = _mm_setr_ps(0.119973198f32, 0.0428578928f32, 0.225254923f32, 27f32);
    //     let src = _mm_setr_ps(0.0428578928f32, 0.0428578928f32, 0.0428578928f32, 27f32);
    //     let rgb = Rgb::<u8>::new(0, 0, 0);
    //     let xyz = Xyz::from_srgb(&rgb);
    //     let ln = _mm_cbrt_ps(src);
    //     println!("X: {}, Y: {}, Z: {}", 0.119973198f32.cbrt(), 0.0428578928f32.cbrt(), 0.225254923f32.cbrt());
    //     _mm_storeu_ps(dst.as_mut_ptr() as *mut f32, ln);
    //     println!("{:?}", dst);
    // }
    // #[cfg(target_arch = "aarch64")]
    // unsafe {
    //     let m = vdupq_n_f32(27f32);
    //     let cbrt = vcbrtq_f32_ulp2(m);
    //     let l = vgetq_lane_f32::<0>(cbrt);
    //     println!("Cbrt {}", l);
    // }

    let r = 140;
    let g = 164;
    let b = 177;
    let rgb = Rgb::<u8>::new(r, g, b);
    let hsl = rgb.to_hsl();
    println!("RGB {:?}", rgb);
    println!("HSL {:?}", hsl);
    println!("Back RGB {:?}", hsl.to_rgb8());

    // unsafe {
    //     let (h, s, l) = neon_rgb_to_hsl(
    //         vdupq_n_u32(r as u32),
    //         vdupq_n_u32(g as u32),
    //         vdupq_n_u32(b as u32),
    //         vdupq_n_f32(1f32),
    //     );
    //     println!(
    //         "NEON HSL {}, {}, {}",
    //         vgetq_lane_f32::<0>(h),
    //         vgetq_lane_f32::<0>(s),
    //         vgetq_lane_f32::<0>(l)
    //     );
    //     let (r1, g1, b1) = neon_hsl_to_rgb(h, s, l, vdupq_n_f32(1f32));
    //
    //     println!(
    //         "NEON HSL -> RGB {}, {}, {}",
    //         vgetq_lane_u32::<0>(r1),
    //         vgetq_lane_u32::<0>(g1),
    //         vgetq_lane_u32::<0>(b1)
    //     );
    // }
    //
    // unsafe {
    //     let (h, s, v) = neon_rgb_to_hsv(
    //         vdupq_n_u32(r as u32),
    //         vdupq_n_u32(g as u32),
    //         vdupq_n_u32(b as u32),
    //         vdupq_n_f32(1f32),
    //     );
    //     let hsv = rgb.to_hsv();
    //     println!("HSV {:?}", hsv);
    //     println!("HSV->RBB {:?}", hsv.to_rgb8());
    //     println!(
    //         "NEON HSV {}, {}, {}",
    //         vgetq_lane_f32::<0>(h),
    //         vgetq_lane_f32::<0>(s),
    //         vgetq_lane_f32::<0>(v)
    //     );
    //     let (r1, g1, b1) = neon_hsv_to_rgb(h, s, v, vdupq_n_f32(1f32));
    //     println!(
    //         "NEON RGB {}, {}, {}",
    //         vgetq_lane_u32::<0>(r1),
    //         vgetq_lane_u32::<0>(g1),
    //         vgetq_lane_u32::<0>(b1)
    //     );
    // }

    let img = ImageReader::open("./assets/asset_middle.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    println!("dimensions {:?}", img.dimensions());

    println!("{:?}", img.color());
    let mut src_bytes = img.as_bytes();
    let width = dimensions.0;
    let height = dimensions.1;
    let components = 3;

    let mut dst_rgba = vec![];
    dst_rgba.resize(4usize * width as usize * height as usize, 0u8);
    rgb_to_rgba(
        &src_bytes,
        3u32 * width,
        &mut dst_rgba,
        4u32 * width,
        width,
        height,
        255,
    );
    src_bytes = &dst_rgba;

    let mut dst_slice: Vec<u8> = Vec::new();
    dst_slice.resize(width as usize * components * height as usize, 0u8);

    {
        let mut lab_store: Vec<u16> = vec![];
        let store_stride = width as usize * components * std::mem::size_of::<u16>();
        lab_store.resize(width as usize * components * height as usize, 0u16);
        let src_stride = width * components as u32;
        let start_time = Instant::now();
        rgb_to_hsv(
            src_bytes,
            src_stride,
            &mut lab_store,
            store_stride as u32,
            width,
            height,
            100f32,
        );
        let elapsed_time = start_time.elapsed();
        // Print the elapsed time in milliseconds
        println!("RGBA To HSV: {:.2?}", elapsed_time);
        // let mut destination: Vec<f32> = vec![];
        // destination.resize(width as usize * height as usize * 4, 0f32);
        // let dst_stride = width * 4 * std::mem::size_of::<f32>() as u32;
        // append_alpha(&mut destination, dst_stride, &store, store_stride as u32, &alpha_store, alpha_stride as u32, width, height);

        let lab_stride = width as usize * 3usize * std::mem::size_of::<f32>();
        //
        // let mut src_shift = 0usize;
        // for _ in 0..height as usize {
        //     let src_ptr = unsafe { (src.as_ptr() as *mut u8).add(src_shift) as *mut f32 };
        //     let src_slice = unsafe { slice::from_raw_parts(src_ptr, width as usize * 4) };
        //
        //     for x in 0..width as usize {
        //         let px = x * 4;
        //         lab_store[px] = src_slice[px];
        //         lab_store[px + 1] = src_slice[px + 1];
        //         lab_store[px + 2] = src_slice[px + 2];
        //         a_store[x] = src_slice[px + 3];
        //     }
        //     src_shift += src_stride as usize;
        // }

        let start_time = Instant::now();
        hsv_to_rgb(
            &lab_store,
            store_stride as u32,
            &mut dst_slice,
            src_stride,
            width,
            height,
            100f32,
        );

        let elapsed_time = start_time.elapsed();
        // Print the elapsed time in milliseconds
        println!("HSV To RGBA: {:.2?}", elapsed_time);

        // laba_to_srgb(
        //     &lab_store,
        //     lab_stride as u32,
        //     &alpha_store,
        //     width * std::mem::size_of::<f32>() as u32,
        //     &mut dst_slice,
        //     width * 4,
        //     width,
        //     height,
        // );
        //
        src_bytes = &dst_slice;
    }

    // let mut xyz: Vec<f32> = vec![];
    // xyz.resize(4 * width as usize * height as usize, 0f32);
    //
    // let mut a_plane: Vec<f32> = vec![];
    // a_plane.resize(width as usize * height as usize, 0f32);
    //
    // let mut dst_bytes: Vec<u8> = vec![];
    // dst_bytes.resize(width as usize * components as usize * height as usize, 0u8);
    //
    // let start_time = Instant::now();
    // xyz_to_srgb(
    //     &xyz,
    //     width * 3 * std::mem::size_of::<f32>() as u32,
    //     &mut dst_bytes,
    //     width * components,
    //     width,
    //     height,
    // );
    //
    // linear_to_rgba(
    //     &xyz,
    //     width * 4 * std::mem::size_of::<f32>() as u32,
    //     &mut dst_bytes,
    //     width * components,
    //     width,
    //     height,
    //     TransferFunction::Srgb,
    // );

    // linear_to_rgb(
    //     &xyz,
    //     width * 3 * std::mem::size_of::<f32>() as u32,
    //     &mut dst_bytes,
    //     width * components,
    //     width,
    //     height,
    //     TransferFunction::Srgb,
    // );

    // let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    // println!("XYZ to sRGB: {:.2?}", elapsed_time);

    // let rgba = rgb_to_rgba(&dst_bytes, width, height);

    if components == 4 {
        image::save_buffer(
            "converted.png",
            src_bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgba8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "converted.jpg",
            src_bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    }
}
