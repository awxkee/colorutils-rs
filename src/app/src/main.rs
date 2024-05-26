use colorutils_rs::*;
use image::io::Reader as ImageReader;
use image::{EncodableLayout, GenericImageView};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

fn rgb_to_rgba(vec: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.resize(height as usize * width as usize * 4usize, 0u8);
    let src_stride = width as usize * 3;
    let dst_stride = width as usize * 4;
    for y in 0..height as usize {
        for x in 0..width as usize {
            bytes[dst_stride * y + x * 4] = vec[src_stride * y + x * 3];
            bytes[dst_stride * y + x * 4 + 1] = vec[src_stride * y + x * 3 + 1];
            bytes[dst_stride * y + x * 4 + 2] = vec[src_stride * y + x * 3 + 2];
            bytes[dst_stride * y + x * 4 + 3] = 255;
        }
    }
    bytes
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
    //     let cbrt = vexpq_f32(m);
    //     let l = vgetq_lane_f32::<0>(cbrt);
    //     println!("Exp {}", l);
    // }

    let img = ImageReader::open("./assets/asset.jpg")
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

    // let rgba = &rgb_to_rgba(&src_bytes, width, height);
    // src_bytes = rgba;

    let mut xyz: Vec<f32> = vec![];
    xyz.resize(3 * width as usize * height as usize, 0f32);

    let mut a_plane: Vec<f32> = vec![];
    a_plane.resize(width as usize * height as usize, 0f32);

    for i in 0..10 {
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
        rgb_to_lab(
            src_bytes,
            width * components,
            &mut xyz,
            width * 3 * std::mem::size_of::<f32>() as u32,
            width,
            height,
        );
        // rgb_to_linear(
        //     src_bytes,
        //     width * components,
        //     &mut xyz,
        //     width * 3 * std::mem::size_of::<f32>() as u32,
        //     width,
        //     height,
        //     TransferFunction::Srgb,
        // );
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

    lab_to_srgb(
        &xyz,
        width * 3 * std::mem::size_of::<f32>() as u32,
        &mut dst_bytes,
        width * components,
        width,
        height,
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
