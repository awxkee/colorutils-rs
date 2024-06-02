use colorutils_rs::*;
use image::io::Reader as ImageReader;
use image::{EncodableLayout, GenericImageView};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
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
    //     let m = vdupq_n_f32(std::f32::consts::E);
    //     let cbrt = vlogq_f32_ulp35(m);
    //     let l = vgetq_lane_f32::<0>(cbrt);
    //     println!("Exp {}", l);
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
    let components = 4;

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
    dst_slice.resize(width as usize * 4 * height as usize, 0u8);

    {
        let mut lab_store: Vec<f32> = vec![];
        let store_stride = width as usize * 4usize * std::mem::size_of::<f32>();
        lab_store.resize(width as usize * 4usize * height as usize, 0f32);
        let mut alpha_store: Vec<f32> = vec![];
        let alpha_stride = width as usize * std::mem::size_of::<f32>();
        alpha_store.resize(width as usize * height as usize, 0f32);
        rgba_to_lab_with_alpha(
            src_bytes,
            4u32 * width,
            &mut lab_store,
            store_stride as u32,
            width,
            height,
        );
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

        lab_with_alpha_to_rgba(
            &lab_store,
            store_stride as u32,
            &mut dst_slice,
            4u32 * width,
            width,
            height,
        );

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

    let mut xyz: Vec<f32> = vec![];
    xyz.resize(4 * width as usize * height as usize, 0f32);

    let mut a_plane: Vec<f32> = vec![];
    a_plane.resize(width as usize * height as usize, 0f32);

    for i in 0..1 {
        let start_time = Instant::now();
        // srgba_to_xyza(
        //     src_bytes,
        //     width * components,
        //     &mut xyz,
        //     width * 3 * std::mem::size_of::<f32>() as u32,
        //     &mut a_plane,
        //     width as u32 * std::mem::size_of::<f32>() as u32,
        //     width,
        //     height,
        // );
        // rgba_to_linear(
        //     src_bytes,
        //     width * components,
        //     &mut xyz,
        //     width * 3 * std::mem::size_of::<f32>() as u32,
        //     width,
        //     height,
        //     TransferFunction::Srgb,
        // );
        rgba_to_linear(
            src_bytes,
            width * components,
            &mut xyz,
            width * 4 * std::mem::size_of::<f32>() as u32,
            width,
            height,
            TransferFunction::Srgb,
        );
        let elapsed_time = start_time.elapsed();
        // Print the elapsed time in milliseconds
        println!("sRGB to XYZ: {:.2?}", elapsed_time);
    }

    let mut dst_bytes: Vec<u8> = vec![];
    dst_bytes.resize(width as usize * components as usize * height as usize, 0u8);

    let start_time = Instant::now();
    // xyz_to_srgb(
    //     &xyz,
    //     width * 3 * std::mem::size_of::<f32>() as u32,
    //     &mut dst_bytes,
    //     width * components,
    //     width,
    //     height,
    // );

    linear_to_rgba(
        &xyz,
        width * 4 * std::mem::size_of::<f32>() as u32,
        &mut dst_bytes,
        width * components,
        width,
        height,
        TransferFunction::Srgb,
    );

    // linear_to_rgb(
    //     &xyz,
    //     width * 3 * std::mem::size_of::<f32>() as u32,
    //     &mut dst_bytes,
    //     width * components,
    //     width,
    //     height,
    //     TransferFunction::Srgb,
    // );

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("XYZ to sRGB: {:.2?}", elapsed_time);

    // let rgba = rgb_to_rgba(&dst_bytes, width, height);

    if components == 4 {
        image::save_buffer(
            "converted.png",
            dst_bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgba8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "converted.jpg",
            dst_bytes.as_bytes(),
            dimensions.0,
            dimensions.1,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    }
}
