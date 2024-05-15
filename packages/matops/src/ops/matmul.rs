use std::ops::{Add, Div, Mul, Sub};

use wasm_bindgen::prelude::*;

use super::meta::NdArrayMetadata;

fn matmul<T: Copy + Add + Sub + Mul + Div + std::iter::Sum<<T as std::ops::Mul>::Output>>(
    a: &[T],
    am: JsValue,
    b: &[T],
    bm: JsValue,
) -> Vec<T> {
    let am = NdArrayMetadata::from(am);
    let bm = NdArrayMetadata::from(bm);
    if am.shape.len() != bm.shape.len() || am.shape.len() > 2 {
        todo!("Not implemented for matrices with more than 2 dimensions");
    }
    let size = am.shape[0] * bm.shape[1];
    let mut buffer = Vec::with_capacity(size);
    for i in 0..am.shape[0] {
        for j in 0..bm.shape[1] {
            let sum = (0..am.shape[1])
                .map(|k| a[i * am.shape[1] + k] * b[k * bm.shape[1] + j])
                .sum();
            buffer.push(sum);
        }
    }
    buffer
}

#[wasm_bindgen]
pub fn matmul_f32(a: &[f32], am: JsValue, b: &[f32], bm: JsValue) -> Vec<f32> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_f64(a: &[f64], am: JsValue, b: &[f64], bm: JsValue) -> Vec<f64> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_i8(a: &[i8], am: JsValue, b: &[i8], bm: JsValue) -> Vec<i8> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_i16(a: &[i16], am: JsValue, b: &[i16], bm: JsValue) -> Vec<i16> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_i32(a: &[i32], am: JsValue, b: &[i32], bm: JsValue) -> Vec<i32> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_i64(a: &[i64], am: JsValue, b: &[i64], bm: JsValue) -> Vec<i64> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_u8(a: &[u8], am: JsValue, b: &[u8], bm: JsValue) -> Vec<u8> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_u16(a: &[u16], am: JsValue, b: &[u16], bm: JsValue) -> Vec<u16> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_u32(a: &[u32], am: JsValue, b: &[u32], bm: JsValue) -> Vec<u32> {
    matmul(a, am, b, bm)
}

#[wasm_bindgen]
pub fn matmul_u64(a: &[u64], am: JsValue, b: &[u64], bm: JsValue) -> Vec<u64> {
    matmul(a, am, b, bm)
}
