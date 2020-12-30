use criterion::{criterion_group, criterion_main, Criterion};

use ndarray::{arr2, Array2};
use rand::Rng;
use rand_pcg::Mcg128Xsl64;

use rust_nn::{train::*, Float};

const BATCH_SIZE: usize = 128;
const INPUT_SIZE: usize = 12;

fn make_input() -> (Array2<Float>, Array2<Float>) {
    let mut random = Mcg128Xsl64::new(1);
    let mut x = Vec::new();
    let mut t = Vec::new();
    for _ in 0..BATCH_SIZE {
        let mut row = [0.0; INPUT_SIZE];
        for r in row.iter_mut() {
            *r = random.gen_range(-3.0..=3.0);
        }
        x.push(row);
        t.push([random.gen_range(-3.0..=3.0)]);
    }
    (arr2(&x), arr2(&t))
}

fn nn2_16_sgd(c: &mut Criterion) {
    let mut nn = NN2Regression::new(
        [INPUT_SIZE, 16, 16],
        BATCH_SIZE,
        SGD::default(),
        SGD::default(),
    );
    let (x, t) = make_input();
    c.bench_function("nn2_16_sgd", |b| {
        b.iter(|| {
            nn.train(&x, &t);
        });
    });
}

fn nn2_32_sgd(c: &mut Criterion) {
    let mut nn = NN2Regression::new(
        [INPUT_SIZE, 32, 32],
        BATCH_SIZE,
        SGD::default(),
        SGD::default(),
    );
    let (x, t) = make_input();
    c.bench_function("nn2_32_sgd", |b| {
        b.iter(|| {
            nn.train(&x, &t);
        });
    });
}

fn nn2_16_adam(c: &mut Criterion) {
    let mut nn = NN2Regression::new(
        [INPUT_SIZE, 16, 16],
        BATCH_SIZE,
        Adam::default(),
        Adam::default(),
    );
    let (x, t) = make_input();
    c.bench_function("nn2_16_adam", |b| {
        b.iter(|| {
            nn.train(&x, &t);
        });
    });
}

fn nn2_32_adam(c: &mut Criterion) {
    let mut nn = NN2Regression::new(
        [INPUT_SIZE, 32, 32],
        BATCH_SIZE,
        Adam::default(),
        Adam::default(),
    );
    let (x, t) = make_input();
    c.bench_function("nn2_32_adam", |b| {
        b.iter(|| {
            nn.train(&x, &t);
        });
    });
}

criterion_group!(benches, nn2_16_sgd, nn2_32_sgd, nn2_16_adam, nn2_32_adam);
criterion_main!(benches);
