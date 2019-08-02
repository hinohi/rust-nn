#![feature(test)]

extern crate test;
use test::Bencher;

use ndarray::{arr2, Array2};
use rand_pcg::Mcg128Xsl64;

use rand::Rng;
use rust_nn::train::{Adam, NN2Regression, NN4Regression, SGD};
use rust_nn::Float;

const BATCH_SIZE: usize = 100;
const INPUT_SIZE: usize = 8;

fn make_input() -> (Array2<Float>, Array2<Float>) {
    let mut random = Mcg128Xsl64::new(1);
    let mut x = Vec::new();
    let mut t = Vec::new();
    for _ in 0..BATCH_SIZE {
        let mut row = [0.0; INPUT_SIZE];
        for r in row.iter_mut() {
            *r = random.gen_range(-3.0, 3.0);
        }
        x.push(row);
        t.push([random.gen_range(-3.0, 3.0)]);
    }
    (arr2(&x), arr2(&t))
}

#[bench]
fn nn2_sgd(b: &mut Bencher) {
    let mut nn = NN2Regression::new(
        [INPUT_SIZE, 16, 16],
        BATCH_SIZE,
        SGD::default(),
        SGD::default(),
    );
    let (x, t) = make_input();
    b.iter(|| {
        nn.train(&x, &t);
    });
}

#[bench]
fn nn4_sgd(b: &mut Bencher) {
    let mut nn = NN4Regression::new(
        [INPUT_SIZE, 32, 32, 32, 32],
        BATCH_SIZE,
        SGD::default(),
        SGD::default(),
    );
    let (x, t) = make_input();
    b.iter(|| {
        nn.train(&x, &t);
    });
}

#[bench]
fn nn2_adam(b: &mut Bencher) {
    let mut nn = NN2Regression::new(
        [INPUT_SIZE, 16, 16],
        BATCH_SIZE,
        Adam::default(),
        Adam::default(),
    );
    let (x, t) = make_input();
    b.iter(|| {
        nn.train(&x, &t);
    });
}

#[bench]
fn nn4_adam(b: &mut Bencher) {
    let mut nn = NN4Regression::new(
        [INPUT_SIZE, 32, 32, 32, 32],
        BATCH_SIZE,
        Adam::default(),
        Adam::default(),
    );
    let (x, t) = make_input();
    b.iter(|| {
        nn.train(&x, &t);
    });
}
