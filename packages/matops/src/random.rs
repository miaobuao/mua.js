use rand::{rngs::ThreadRng, Rng};
use wasm_bindgen::prelude::*;

pub struct NormalRandomGenerater {
    mean: f32,
    std: f32,
    rng: ThreadRng,
}

impl NormalRandomGenerater {
    pub fn new(mean: Option<f32>, std: Option<f32>) -> Self {
        let mean = mean.unwrap_or(0.0);
        let std = std.unwrap_or(1.0);
        NormalRandomGenerater {
            mean,
            std,
            rng: rand::thread_rng(),
        }
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

#[wasm_bindgen]
pub fn normal(len: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut normal = NormalRandomGenerater::new(Some(mean), Some(std));
    let mut buffer = vec![0.0; len];
    buffer.iter_mut().for_each(|x| *x = normal.next().unwrap());
    buffer
}
