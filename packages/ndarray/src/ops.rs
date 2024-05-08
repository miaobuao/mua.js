use std::{borrow::Borrow, vec};
use wasm_bindgen::prelude::*;
use crate::{ndarray::NdArray, utils::nd_idx_to_offset};


#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}



#[wasm_bindgen]
impl NdArray {
    pub fn matmul(&self, b: &NdArray) -> NdArray {
        matmul(self, b)
    }

    pub fn dot(&self, b: &NdArray) -> NdArray {
        dot(self, b)
    }

    #[wasm_bindgen(js_name = mulScalar)]
    pub fn mul_scalar(&self, b: f32) -> NdArray {
        mul_scalar(self, b)
    }

    pub fn add(&self, b: &NdArray) -> NdArray {
        add(self, b)
    }

    #[wasm_bindgen(js_name = addScalar)]
    pub fn add_scalar(&self, b: f32) -> NdArray {
        add_scalar(self, b)
    }

    pub fn sub(&self, b: &NdArray) -> NdArray {
        sub(self, b)
    }

    #[wasm_bindgen(js_name = subScalar)]
    pub fn sub_scalar(&self, b: f32) -> NdArray {
        sub_scalar(self, b)
    }

    pub fn log(&self, base: f32) -> NdArray {
        log(self, base)
    }

    pub fn ln(&self) -> NdArray {
        ln(self)
    }

    pub fn exp(&self) -> NdArray {
        exp(self)
    }

    pub fn relu(&self) -> NdArray {
        relu(self)
    }

    pub fn sigmoid(&self) -> NdArray {
        sigmoid(self)
    }

    pub fn tanh(&self) -> NdArray {
        tanh(self)
    }

    pub fn softmax(&self, dim: Option<usize>) -> NdArray {
        softmax(self, dim)
    }

    pub fn pow(&self, b: f32) -> NdArray {
        pow(self, b)
    }

    pub fn padding_1d(&self, value: f32, px: usize, py: usize) -> NdArray {
        let mut res = self.clone();
        if px > 0 {
            res = pad_x_1d(&res, px, Some(value));
        }
        if py > 0 {
            res = pad_y_2d(&res, py, Some(value));
        }
        res
    }

    pub fn sum(&self) -> f32 {
        self.buffer.iter().sum()
    }
}

#[wasm_bindgen]
pub fn matmul(a: &NdArray, b: &NdArray) -> NdArray {
    if a.shape.len() != b.shape.len() || a.shape.len() > 2 {
        todo!("Not implemented for matrices with more than 2 dimensions");
    }
    let size = a.shape[0] * b.shape[1];
    let mut buffer = Vec::with_capacity(size);
    for i in 0..a.shape[0] {
        for j in 0..b.shape[1] {
            let mut sum = 0.0;
            for k in 0..a.shape[1] {
                sum += a.buffer[i * a.shape[1] + k] * b.buffer[k * b.shape[1] + j];
            }
            buffer.push(sum);
        }
    }
    NdArray::from(&buffer, Some(vec![a.shape[0], b.shape[1]]), None)
}

#[wasm_bindgen(js_name = mulScalar)]
pub fn mul_scalar(a: &NdArray, b: f32) -> NdArray {
    let mut c = a.clone();
    for i in 0..c.buffer.len() {
        c.buffer[i] = a.buffer[i] * b;
    }
    c
}

#[wasm_bindgen]
pub fn dot(a: &NdArray, b: &NdArray) -> NdArray {
    let mut c = a.clone();
    for i in 0..c.buffer.len() {
        c.buffer[i] = a.buffer[i] * b.buffer[i];
    }
    c
}

#[wasm_bindgen]
pub fn add(a: &NdArray, b: &NdArray) -> NdArray {
    let mut c = a.clone();
    if a.buffer.len() % b.buffer.len() != 0 {
        todo!("Not implemented for matrices with unequal shapes");
    }
    for i in 0..c.buffer.len() {
        c.buffer[i] = a.buffer[i] + b.buffer[i % b.buffer.len()];
    }
    c
}

#[wasm_bindgen(js_name = addScalar)]
pub fn add_scalar(a: &NdArray, b: f32) -> NdArray {
    let mut c = a.clone();
    for i in 0..c.buffer.len() {
        c.buffer[i] = a.buffer[i] + b;
    }
    c
}

#[wasm_bindgen]
pub fn sub(a: &NdArray, b: &NdArray) -> NdArray {
    let mut c = a.clone();
    for i in 0..c.buffer.len() {
        c.buffer[i] = a.buffer[i] - b.buffer[i];
    }
    c
}

#[wasm_bindgen(js_name = subScalar)]
pub fn sub_scalar(a: &NdArray, b: f32) -> NdArray {
    let mut c = a.clone();
    for i in 0..c.buffer.len() {
        c.buffer[i] = a.buffer[i] - b;
    }
    c
}

#[wasm_bindgen]
pub fn log(a: &NdArray, base: f32) -> NdArray {
    let mut res = a.clone();
    for i in 0..res.buffer.len() {
        res.buffer[i] = f32::log(res.buffer[i], base);
    }
    res
}

#[wasm_bindgen]
pub fn ln(a: &NdArray) -> NdArray {
    let mut res = a.clone();
    for i in 0..res.buffer.len() {
        res.buffer[i] = f32::ln(res.buffer[i]);
    }
    res
}

#[wasm_bindgen]
pub fn exp(a: &NdArray) -> NdArray {
    let mut res = a.clone();
    for i in 0..res.buffer.len() {
        res.buffer[i] = f32::exp(res.buffer[i]);
    }
    res
}

#[wasm_bindgen]
pub fn relu(a: &NdArray) -> NdArray {
    let mut res = a.clone();
    for i in 0..res.buffer.len() {
        res.buffer[i] = f32::max(0.0, res.buffer[i]);
    }
    res
}

#[wasm_bindgen]
pub fn sigmoid(a: &NdArray) -> NdArray {
    let mut res = a.clone();
    for i in 0..res.buffer.len() {
        res.buffer[i] = 1.0 / (1.0 + f32::exp(-res.buffer[i]));
    }
    res
}

#[wasm_bindgen]
pub fn tanh(a: &NdArray) -> NdArray {
    let mut res = a.clone();
    for i in 0..res.buffer.len() {
        res.buffer[i] = f32::tanh(res.buffer[i]);
    }
    res
}


#[wasm_bindgen]
pub fn softmax(a: &NdArray, dim: Option<usize>) -> NdArray {
    fn softmax_internal(buffer: &mut [f32]) {
        let sum  = buffer.iter().sum::<f32>();
        buffer.iter_mut().for_each(|x| *x /= sum);
    }

    fn softmax_last_dim(buffer: &mut [f32],  chunk_size: usize) {
       buffer.chunks_mut(chunk_size).for_each(softmax_internal);
    }
    
    let mut res = a.exp();
    if let Some(dim) = dim {
        if dim < res.shape.len() -1 {
            let stride = a.strides[dim];
            for i in 0..stride {
                let mut sum = 0.;
                let mut ofst = i;
                while ofst < res.buffer.len() {
                    sum += res.buffer[ofst];
                    ofst += stride;
                }
                ofst = i;
                while ofst < res.buffer.len() {
                    res.buffer[ofst] /= sum;
                    ofst += stride;
                }
            }
            return  res;
        } else {
           softmax_last_dim(&mut res.buffer, *res.shape.last().unwrap());
           return  res;
        }
    } else {
        if a.shape.len() == 1 {
            softmax_internal(&mut res.buffer);
            return  res;
        }else {
            // default: dim is -1
            softmax_last_dim(&mut res.buffer, *res.shape.last().unwrap());
            return res;
        }
    }
}

#[wasm_bindgen]
pub fn pow(a: &NdArray, b: f32) -> NdArray {
    let mut res = a.clone();
    for i in 0..res.buffer.len() {
        res.buffer[i] = f32::powf(res.buffer[i], b);
    }
    res
}

#[wasm_bindgen(js_name = padY2D)]
pub fn pad_y_2d(a: &NdArray, size: usize, value: Option<f32>) -> NdArray {
    if size == 0 {
        return a.clone();
    }
    let mut shape = a.shape.clone();
    shape[a.shape.len() - 2] += size * 2;
    let pad = vec![value.unwrap_or(0.0); a.shape[a.shape.len() - 1]];
    let mut buffer = Vec::new();
    let chunk_size = a.shape.iter().rev().take(2).product();
    for chunk in a.buffer.chunks(chunk_size) {
        for _ in 0..size {
            buffer.extend_from_slice(pad.borrow());
        }
        buffer.extend_from_slice(chunk);
        for _ in 0..size {
            buffer.extend_from_slice(pad.borrow());
        }
    }
    NdArray::from(&buffer, Some(shape), None)
}

/**
 * a: [H, W]
 */
#[wasm_bindgen(js_name = padX1D)]
pub fn pad_x_1d(a: &NdArray, size: usize, value: Option<f32>) -> NdArray {
    if size == 0 {
        return a.clone();
    }
    let mut shape = a.shape.clone();
    shape[a.shape.len() - 1] += size * 2;
    let pad = vec![value.unwrap_or(0.0); size];
    let mut buffer = Vec::new();
    let chunk_size = a.shape[a.shape.len() - 1];
    for chunk in a.buffer.chunks(chunk_size) {
        buffer.extend_from_slice(pad.borrow());
        buffer.extend_from_slice(chunk);
        buffer.extend_from_slice(pad.borrow());
    }
    NdArray::from(&buffer, Some(shape), None)
}

#[wasm_bindgen]
pub fn im2col(
    x: &NdArray,
    kernel_size: &[usize],
    stride: usize,
    padding: &[usize],
    pad_value: Option<f32>,
) -> NdArray {
    if padding.len() > 2 {
        panic!("padding.length must be 0 or 2")
    }
    let mut x = x.to_owned();
    if let Some(&px) = padding.get(0) {
        x = pad_x_1d(&x, px, pad_value);
    }
    // if let Some(&py) = padding.get(1) {
    //     x = pad_y_1d(&x, py, pad_value);
    // }
    dbg!(&x.shape);
    let mut res = Vec::new();
    let chunk_size = kernel_size.iter().product::<usize>() * x.shape.last().unwrap();
    match x.shape.len() {
        2 => {
            // * for conv1d
            let mut cnt = 0;
            let step = stride * x.shape[1];
            // dbg!(&step);
            for i in 0..x.buffer.len() / step {
                let offset = i * step;
                if offset + chunk_size > x.buffer.len() {
                    continue;
                }
                // dbg!(&offset);
                let chunk = x.buffer[offset..offset + chunk_size].to_vec();
                assert_eq!(chunk.len(), chunk_size);
                res.extend_from_slice(chunk.borrow());
                cnt += 1;
            }
            NdArray::from(&res, Some(vec![cnt, chunk_size]), None)
        }
        3 => {
            // * for conv2d
            //      x: [H, W, C]
            //      k: [k, k]
            let mut cnt = 0;
            for i in 0..x.shape[0] / stride {
                let r = i * stride;
                if r + kernel_size[0] > x.shape[0] {
                    continue;
                }
                for j in 0..x.shape[1] / stride {
                    let c = j * stride;
                    if c + kernel_size[1] > x.shape[1] {
                        continue;
                    }
                    println!("r:{}, c:{}", r, c);
                    let mut chunk = Vec::new();
                    for k in 0..kernel_size[0] {
                        let start_ofst = nd_idx_to_offset(&[r + k, c, 0], &x.strides);
                        let end_ofst =
                            nd_idx_to_offset(&[r + k, c + kernel_size[1], 0], &x.strides);
                        if end_ofst > x.buffer.len() {
                            break;
                        }
                        let slice = &x.buffer[start_ofst..end_ofst];
                        chunk.extend_from_slice(slice);
                    }
                    if chunk.len() != chunk_size {
                        continue;
                    }
                    cnt += 1;
                    res.extend_from_slice(chunk.borrow());
                }
            }
            NdArray::from(&res, Some(vec![cnt, chunk_size]), None)
        }
        _ => todo!(),
    }
}

#[test]
fn test_add() {
    let a = NdArray::zeros(&[2, 2, 2]);
    let b = NdArray::arange(0, 4, None).reshape(&[2, 2]);
    let c = a.add(&b);
    assert_eq!(c.buffer, vec![0., 1., 2., 3., 0., 1., 2., 3.]);
}

#[test]
fn test_matmul() {
    let a = NdArray::from(&[1., 2., 3., 4., 5., 6.], Some(vec![2, 3]), None);
    let b = a.reshape(&[3, -1]);
    let c = a.matmul(&b);
    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(c.buffer, vec![22., 28., 49., 64.]);
}

#[test]
fn test_softmax() {
    let a = NdArray::from(&[1., 2., 3., 4.], Some(vec![4]), None);
    let b = softmax(&a, None);
    assert_eq!(
        b.buffer,
        vec![0.032058604, 0.08714432, 0.23688282, 0.6439142]
    );
    let c = a.reshape(&[2, 2]);
    let d = softmax(&c, Some(0));
    assert_eq!(d.buffer, vec![0.11920293, 0.11920293, 0.88079715, 0.8807971]);
    let e = softmax(&c, Some(1));
    assert_eq!(e.buffer, vec![0.2689414, 0.7310586, 0.26894143, 0.7310586]);
}

// #[test]
// fn test_pad_y() {
//     let a = NdArray::from(&[1., 2., 3., 4.], Some(vec![2, 2]), None);
//     let b = pad_y_1d(&a, 1, None);
//     assert_eq!(b.buffer, vec![0., 0., 1., 2., 3., 4., 0., 0.]);
//     assert_eq!(b.shape, vec![4, 2]);

//     let c = a.reshape(&[1, 4]);
//     let d = pad_y_1d(&c, 2, Some(1.));
//     assert_eq!(
//         d.buffer,
//         vec![1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1., 1., 1.,]
//     );
// }

#[test]
fn test_pad_x() {
    let a = NdArray::from(&[1., 2., 3., 4.], Some(vec![2, 2]), None);
    let b = pad_x_1d(&a, 1, None);
    assert_eq!(b.shape, vec![2, 4]);
    assert_eq!(b.buffer, vec![0., 1., 2., 0., 0., 3., 4., 0.]);

    let c = a.reshape(&[1, 4]);
    let d = pad_x_1d(&c, 2, Some(1.));
    assert_eq!(d.buffer, vec![1., 1., 1., 2., 3., 4., 1., 1.,]);
}

#[test]
fn test_im2col() {
    // ---- 1d
    // -------- normal
    let seq = NdArray::arange(0, 5 * 3, None).reshape(&[5, 3]);
    let col = im2col(&seq, &[3], 1, &[0], None);
    assert_eq!(col.shape, vec![3, 9]);
    assert_eq!(
        col.buffer,
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0
        ]
    );
    // -------- stride
    let col = im2col(&seq, &[3], 2, &[0], None);
    assert_eq!(col.shape, vec![2, 9]);
    assert_eq!(
        col.buffer,
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0
        ]
    );
    // -------- padding
    // let col = im2col(&seq, &[3], 2, &[1], None);
    // assert_eq!(col.shape, vec![3, 9]);
    // assert_eq!(
    //     col.buffer,
    //     vec![
    //         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0,
    //         6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0
    //     ]
    // );

    // ---- 2d
    let img = NdArray::arange(0, 28 * 28 * 3, None).reshape(&[28,28, 3]);
    let col = im2col(&img, &[3, 3], 1, &[0, 0], None);
    assert_eq!(col.shape, vec![26 * 26, 3 * 3 * 3]);
    
    let col = im2col(&img, &[3, 3], 2, &[0, 0], None);
    assert_eq!(col.shape, vec![13 * 13, 3 * 3 * 3]);
}
