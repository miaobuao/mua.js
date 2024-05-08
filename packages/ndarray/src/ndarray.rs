use std::{borrow::Borrow, vec};

use crate::utils::{
    self, get_indexes, get_strides, nd_idx_to_offset, reorder, NormalRandomGenerater,
};
use wasm_bindgen::prelude::*;

use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

#[wasm_bindgen]
pub struct NdArray {
    pub(super) buffer: Vec<f32>,
    pub(super) strides: Vec<usize>,
    pub(super) shape: Vec<usize>,
}

#[wasm_bindgen]
impl NdArray {
    pub fn from(buffer: &[f32], shape: Option<Vec<usize>>, strides: Option<Vec<usize>>) -> Self {
        let shape = shape.unwrap_or(vec![buffer.len()]);
        let strides = strides.unwrap_or(utils::get_strides(shape.borrow()));
        Self {
            buffer: buffer.to_vec(),
            strides: strides,
            shape: shape,
        }
    }

    pub fn arange(start: usize, stop: usize, step: Option<usize>) -> Self {
        Self::from(
            (start..stop)
                .step_by(step.unwrap_or(1))
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
                .as_slice(),
            None,
            None,
        )
    }

    #[wasm_bindgen(getter, js_name = "buffer")]
    pub fn get_buffer(&self) -> Vec<f32> {
        self.buffer.clone()
    }

    #[wasm_bindgen(getter, js_name = "shape")]
    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn map(&self, f: &js_sys::Function) -> Self {
        Self::from(
            &self
                .buffer
                .iter()
                .map(|x| {
                    f.call1(&JsValue::null(), &JsValue::from(*x))
                        .unwrap()
                        .as_f64()
                        .unwrap() as f32
                })
                .collect::<Vec<f32>>(),
            Some(self.shape.clone()),
            Some(self.strides.clone()),
        )
    }

    pub fn rand(shape: &[usize]) -> Self {
        let mut rng = thread_rng();
        let mut buffer = vec![0.0; shape.iter().product()];
        buffer.iter_mut().for_each(|x| *x = rng.gen_range(0.0..1.0));
        Self::from(&buffer, Some(shape.to_vec()), None)
    }

    #[wasm_bindgen(js_name = randBetween)]
    pub fn rand_between(shape: &[usize], min: f32, max: f32) -> Self {
        let mut rng = thread_rng();
        let mut buffer = vec![0.0; shape.iter().product()];
        buffer.iter_mut().for_each(|x| *x = rng.gen_range(min..max));
        Self::from(&buffer, Some(shape.to_vec()), None)
    }

    pub fn normal(shape: &[usize], mean: f32, std: f32) -> Self {
        let mut normal = NormalRandomGenerater::new(Some(mean), Some(std));
        let mut buffer = vec![0.0; shape.iter().product()];
        buffer.iter_mut().for_each(|x| *x = normal.next().unwrap());
        Self::from(&buffer, Some(shape.to_vec()), None)
    }

    pub fn randn(shape: &[usize]) -> Self {
        NdArray::normal(shape, 0.0, 1.0)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::from(
            &vec![0.0; shape.iter().product()],
            Some(shape.to_vec()),
            None,
        )
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::from(
            &vec![1.0; shape.iter().product()],
            Some(shape.to_vec()),
            None,
        )
    }

    pub fn permute(&self, indices: &[usize]) -> Self {
        assert_eq!(indices.len(), self.shape.len());
        let indices = indices.to_vec();
        let output_shape: Vec<usize> = reorder(&self.shape, &indices);
        let output_strides: Vec<usize> = get_strides(&output_shape);
        let mut output_buffer = self.buffer.clone();
        for input_index in get_indexes(&self.shape) {
            let input_offset = nd_idx_to_offset(&input_index, &self.strides);
            let output_index = reorder(&input_index, &indices);
            let output_offset = nd_idx_to_offset(&output_index, &output_strides);
            output_buffer[output_offset] = self.buffer[input_offset];
        }
        Self {
            buffer: output_buffer,
            shape: output_shape,
            strides: output_strides,
        }
    }

    pub fn reshape(&self, shape: &[i32]) -> Self {
        let new_axis_count = shape.iter().filter(|x| (**x) <= 0).count();
        if new_axis_count == 0 {
            if shape
                .iter()
                .map(|x| *x as usize)
                .reduce(|a, b| a * b)
                .unwrap()
                != self.buffer.len()
            {
                panic!("shape is not compatible with buffer")
            }
            let shape: Vec<usize> = shape.iter().map(|x| *x as usize).collect();
            return Self {
                buffer: self.buffer.clone(),
                strides: utils::get_strides(shape.borrow()),
                shape,
            };
        } else if new_axis_count == 1 {
            let known_size = shape
                .iter()
                .map(|x| *x)
                .reduce(|acc, e| {
                    if e < -1 || e == 0 {
                        panic!("unknown dimension size must be positive or -1.");
                    } else if e == -1 {
                        return acc;
                    }
                    acc * e
                })
                .unwrap();
            let new_size = self.buffer.len() / known_size as usize;
            let shape: Vec<usize> = shape
                .iter()
                .map(|x| {
                    let x = *x;
                    if x == -1 {
                        return new_size;
                    }
                    return x as usize;
                })
                .collect();
            return Self::from(&self.buffer, Some(shape), None);
        }
        panic!("too many new axis");
    }

    pub fn transpose(&self) -> Self {
        let mut axis = (0..self.shape.len()).collect::<Vec<usize>>();
        axis.reverse();
        self.permute(axis.borrow())
    }

    pub fn flatten(&self) -> Self {
        self.reshape(&[self.buffer.len() as i32])
    }

    pub fn slice(&self, indexes: &[usize]) -> Self {
        assert!(indexes.len() <= self.shape.len());
        let remaining = &self.shape[indexes.len()..];
        let offsets: usize = self
            .strides
            .iter()
            .take(indexes.len())
            .enumerate()
            .map(|(idx, stride)| stride * indexes[idx])
            .sum();
        let buf = self.buffer[offsets..offsets + remaining.iter().product::<usize>()].to_vec();

        NdArray::from(
            &buf,
            Some(remaining.iter().map(|x| *x).collect::<Vec<usize>>()),
            None,
        )
    }

    pub fn set(&self, indexes: &[usize], value: NdArray) -> NdArray {
        assert!(indexes.len() <= self.shape.len());
        let remaining = &self.shape[indexes.len()..];
        let offsets: usize = self
            .strides
            .iter()
            .take(indexes.len())
            .enumerate()
            .map(|(idx, stride)| stride * indexes[idx])
            .sum();
        assert_eq!(
            remaining.iter().product::<usize>(),
            value.shape.iter().product()
        );
        let mut res = self.clone();
        res.buffer[offsets..offsets + remaining.iter().product::<usize>()]
            .clone_from_slice(&value.buffer);
        return res;
    }
}

impl Clone for NdArray {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            strides: self.strides.clone(),
            shape: self.shape.clone(),
        }
    }
}

// impl fmt::Display for NdArray {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         fn display(f: &mut std::fmt::Formatter<'_>, v: &[f32], shape: &[usize], deep: usize) {
//             if shape.len() == 1 {
//                 for el in v {
//                     write!(f, "{}, ", el).unwrap();
//                 }
//                 return;
//             }

//             for _ in 0..deep {
//                 write!(f, "\t").unwrap();
//             }

//             write!(f, "[").unwrap();
//             for chunk in v.chunks(shape[0]) {
//                 display(f, &chunk, &shape[1..], deep + 1);
//             }
//             write!(f, "]").unwrap();

//             if deep != 0 {
//                 write!(f, ",\n").unwrap();
//             }
//         }
//         write!(f, "[").unwrap();
//         display(f, &self.buffer, &self.shape, 0);
//         write!(f, "]").unwrap();
//         Ok(())
//     }
// }

#[test]
fn test_random() {
    let a = NdArray::normal(&[100, 20, 10], 0.0, 1.0);
    assert_eq!(a.shape, vec![100, 20, 10]);
}

#[test]
fn test_nd_array_slice() {
    let a = NdArray::arange(0, 12, None).reshape(&[2, 3, 2]);
    let c = a.slice(&[1]);
    assert_eq!(c.shape, vec![3, 2]);
    assert_eq!(c.buffer, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

    let d = NdArray::zeros(&c.shape);
    let e = a.set(&[0], d);
    assert_eq!(e.shape, vec![2, 3, 2]);
    assert_eq!(
        e.buffer,
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    )
}

#[test]
fn test_reshape() {
    let a = NdArray::from(&[1., 2., 3., 4., 5., 6.], Some(vec![2, 3]), None);
    let b = a.reshape(&[3, -1]);
    assert_eq!(b.strides, vec![2, 1]);
    assert_eq!(b.shape, vec![3, 2]);
}

#[test]
fn test_transpose() {
    let a = NdArray::from(&[1., 2., 3., 4., 5., 6.], Some(vec![2, 3]), None);
    assert_eq!(a.strides, vec![3, 1]);

    let b = a.transpose();
    assert_eq!(b.shape, vec![3, 2]);
    assert_eq!(b.strides, vec![2, 1]);
    assert_eq!(b.buffer, vec![1., 4., 2., 5., 3., 6.]);
    assert_eq!(b.transpose().buffer, a.buffer);

    let a = NdArray::from(
        (0..16).map(|x| x as f32).collect::<Vec<f32>>().borrow(),
        Some(vec![2, 4, 2]),
        None,
    );
    let b = a.transpose();
    assert_eq!(
        b.buffer,
        vec![0., 8., 2., 10., 4., 12., 6., 14., 1., 9., 3., 11., 5., 13., 7., 15.]
    )
}

#[test]
fn test_permute() {
    let a = NdArray::from(
        (0..16).map(|x| x as f32).collect::<Vec<f32>>().borrow(),
        Some(vec![2, 4, 2]),
        None,
    );
    let b = a.permute(&[2, 0, 1]);
    assert_eq!(
        b.buffer,
        vec![0., 2., 4., 6., 8., 10., 12., 14., 1., 3., 5., 7., 9., 11., 13., 15.]
    )
}

#[wasm_bindgen]
pub fn concat(a: &NdArray, b: &NdArray) -> NdArray {
    let mut c = a.clone();
    c.buffer.extend_from_slice(&b.buffer);
    c.shape[0] += b.shape[0];
    return c;
}

#[test]
fn test_concat() {
    let a = NdArray::ones(&[2, 3]);
    let b = NdArray::zeros(&[2, 3]);
    let c = concat(&a, &b);
    assert_eq!(c.shape, vec![4, 3]);
    assert_eq!(
        c.buffer,
        vec![1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]
    );
}
