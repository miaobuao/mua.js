use gloo_utils::format::JsValueSerdeExt;
use serde::{Serialize, Deserialize};
use wasm_bindgen::JsValue;

#[derive(Serialize, Deserialize)]
pub struct NdArrayMetadata {
    pub strides: Vec<usize>,
    pub shape: Vec<usize>,
}

impl NdArrayMetadata {
    pub fn from(val: JsValue) -> NdArrayMetadata {
        val.into_serde().unwrap()
    }
}
