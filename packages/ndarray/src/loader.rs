use crate::{ndarray::NdArray, utils::load_image_from_array_buffer};
use std::borrow::Borrow;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = loadImageByRgb)]
pub fn load_image_by_rgb(buffer: &[u8]) -> NdArray {
    if let Some(img) = load_image_from_array_buffer(buffer) {
        let w = img.width() as usize;
        let h = img.height() as usize;

        return NdArray::from(
            img.into_rgb8()
                .iter()
                .map(|x| *x as f32)
                .collect::<Vec<f32>>()
                .borrow(),
            Some(vec![h, w, 3]),
            None,
        );
    }
    todo!()
}

#[wasm_bindgen(js_name = loadImageByRgba)]
pub fn load_image_by_rgba(buffer: &[u8]) -> NdArray {
    if let Some(img) = load_image_from_array_buffer(buffer) {
        let w = img.width() as usize;
        let h = img.height() as usize;
        return NdArray::from(
            img.into_rgba32f()
                .iter()
                .map(|x| *x)
                .collect::<Vec<f32>>()
                .borrow(),
            Some(vec![h, w, 4]),
            None,
        );
    }
    todo!()
}

#[wasm_bindgen(js_name = loadImageByLuma)]
pub fn load_image_by_luma(buffer: &[u8]) -> NdArray {
    if let Some(img) = load_image_from_array_buffer(buffer) {
        let w = img.width() as usize;
        let h = img.height() as usize;
        return NdArray::from(
            img.into_luma16()
                .iter()
                .map(|x| *x as f32)
                .collect::<Vec<f32>>()
                .borrow(),
            Some(vec![h, w, 1]),
            None,
        );
    }
    todo!()
}
