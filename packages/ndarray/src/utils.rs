use rand::rngs::ThreadRng;

use rand::Rng;

pub fn get_strides(shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .map(|(i, _)| shape[i + 1..].iter().product())
        .collect()
}

#[test]
fn test_get_strides() {
    assert_eq!(get_strides(&vec![1, 2, 3]), vec![6, 3, 1]);
    assert_eq!(get_strides(&vec![9, 8, 7]), vec![56, 7, 1]);
}

pub fn get_indexes(shape: &Vec<usize>) -> Vec<Vec<usize>> {
    assert!(shape.len() > 0);
    let mut res: Vec<Vec<usize>> = (0..shape[0]).map(|x| vec![x]).collect();
    for size in shape[1..].iter() {
        let mut new: Vec<Vec<usize>> = Vec::new();
        for cell in res.iter() {
            for value in 0..*size {
                let mut cell = cell.clone();
                cell.push(value);
                new.push(cell);
            }
        }
        res = new;
    }
    res
}

pub struct NormalRandomGenerater {
    mean: f32,
    std: f32,
    rng:  ThreadRng
}

impl NormalRandomGenerater {
    pub fn new(mean: Option<f32>, std: Option<f32>) -> Self {
        let mean = mean.unwrap_or(0.0);
        let std = std.unwrap_or(1.0);
        NormalRandomGenerater { mean, std, rng: rand::thread_rng() }
    }
}

impl Iterator for NormalRandomGenerater {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let u = 1.0 - self.rng.gen::<f32>();
        let v: f32 = self.rng.gen();
        let z = f32::sqrt(-2.0 * f32::ln(u)) * f32::cos(2.0 * std::f32::consts::PI * v);
        Some(z * self.std + self.mean)
    }
}

#[test]
fn test_get_indexes() {
    assert_eq!(
        get_indexes(&vec![3, 2, 3]),
        vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 0, 2],
            vec![0, 1, 0],
            vec![0, 1, 1],
            vec![0, 1, 2],
            vec![1, 0, 0],
            vec![1, 0, 1],
            vec![1, 0, 2],
            vec![1, 1, 0],
            vec![1, 1, 1],
            vec![1, 1, 2],
            vec![2, 0, 0],
            vec![2, 0, 1],
            vec![2, 0, 2],
            vec![2, 1, 0],
            vec![2, 1, 1],
            vec![2, 1, 2],
        ]
    );
}

pub fn reorder<T: Copy>(origin: &Vec<T>, order: &Vec<usize>) -> Vec<T> {
    assert_eq!(origin.len(), order.len());
    let mut target = origin.clone();
    for (src, tgt) in order.iter().enumerate() {
        target[src] = origin[*tgt];
    }
    target
}

#[test]
fn test_reorder_inplace() {
    let  a = vec![2, 4 ,6];
    let b = reorder(&a, &vec![2, 0, 1]);
    assert_eq!(b, [6, 2, 4]);
}

pub fn nd_idx_to_offset(index: &[usize], strides: &Vec<usize>) -> usize {
    assert_eq!(index.len(), strides.len());
    let mut res = 0;
    for (axis, i) in index.iter().enumerate() {
        res += i * strides[axis];
    }
    res
}

#[test]
fn test_nd_idx_to_offset() {
    assert_eq!(nd_idx_to_offset(&vec![1, 5, 1], &vec![3, 2, 3]), 16)
}

