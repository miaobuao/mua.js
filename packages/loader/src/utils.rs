use image::{load_from_memory, DynamicImage};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

pub fn load_image_from_array_buffer(_array: &[u8]) -> Option<DynamicImage> {
    match load_from_memory(_array) {
        Ok(img) => Some(img),
        Err(err) => {
            error(err.to_string().as_str());
            None
        }
    }
}

#[wasm_bindgen]
pub struct ImageHandle {
    buffer: DynamicImage,
    pub width: usize,
    pub height: usize,
}

#[wasm_bindgen]
impl ImageHandle {
    pub fn from(buffer: &[u8]) -> ImageHandle {
        let img = load_image_from_array_buffer(buffer).unwrap();
        let width = img.width() as usize;
        let height = img.height() as usize;
        return ImageHandle {
            buffer: img,
            width,
            height,
        };
    }

    #[wasm_bindgen(getter, js_name=rgb8)]
    pub fn load_image_by_rgb8(&self) -> Vec<u8> {
        return self
            .buffer
            .clone()
            .into_rgba8()
            .iter()
            .map(|x| *x)
            .collect();
    }

    #[wasm_bindgen(getter, js_name=rgba16)]
    pub fn load_image_by_rgba16(&self) -> Vec<u16> {
        return self
            .buffer
            .clone()
            .into_rgba16()
            .iter()
            .map(|x| *x)
            .collect();
    }

    #[wasm_bindgen(getter, js_name=luma16)]
    pub fn load_image_by_luma16(&self) -> Vec<u16> {
        return self
            .buffer
            .clone()
            .into_luma16()
            .iter()
            .map(|x| *x)
            .collect();
    }
}
