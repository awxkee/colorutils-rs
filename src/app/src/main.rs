use std::time::Instant;

use image::{EncodableLayout, GenericImageView};
use image::io::Reader as ImageReader;

use colorutils_rs::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

fn main() {
    let r = 140;
    let g = 164;
    let b = 177;
    let rgb = Rgb::<u8>::new(r, g, b);
    let hsl = rgb.to_hsl();
    println!("RGB {:?}", rgb);
    println!("HSL {:?}", hsl);
    println!("Back RGB {:?}", hsl.to_rgb8());

    let img = ImageReader::open("./assets/beach_horizon.jpg")
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
    //
    // let mut dst_rgba = vec![];
    // dst_rgba.resize(4usize * width as usize * height as usize, 0u8);
    // rgb_to_rgba(
    //     &src_bytes,
    //     3u32 * width,
    //     &mut dst_rgba,
    //     4u32 * width,
    //     width,
    //     height,
    //     255,
    // );
    // src_bytes = &dst_rgba;

    let mut dst_slice: Vec<u8> = Vec::new();
    dst_slice.resize(width as usize * components * height as usize, 0u8);

    {
        let mut lab_store: Vec<f32> = vec![];
        let store_stride = width as usize * components * std::mem::size_of::<f32>();
        lab_store.resize(width as usize * components * height as usize, 0f32);
        let src_stride = width * components as u32;
        let start_time = Instant::now();
        rgb_to_linear(
            src_bytes,
            src_stride,
            &mut lab_store,
            store_stride as u32,
            width,
            height,
            TransferFunction::Gamma2p8,
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
        linear_to_rgb(
            &lab_store,
            store_stride as u32,
            &mut dst_slice,
            src_stride,
            width,
            height,
            TransferFunction::Gamma2p8
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
