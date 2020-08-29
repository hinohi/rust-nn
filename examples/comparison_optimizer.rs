use ndarray::{arr2, Array2};
use rand::Rng;
use rand_pcg::Mcg128Xsl64;
use rust_nn::{train::*, Float};

fn target_func(x: Float, y: Float) -> Float {
    x.sin() * y.cos() * x * y
}

fn gen_case<R: Rng>(batch_size: usize, random: &mut R) -> (Array2<Float>, Array2<Float>) {
    let mut x = Vec::with_capacity(batch_size);
    let mut t = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let a = random.gen_range(-3.0, 3.0);
        let b = random.gen_range(-3.0, 3.0);
        let c = target_func(a, b);
        x.push([a, b]);
        t.push([c]);
    }
    (arr2(&x), arr2(&t))
}

fn main() {
    let mut random = Mcg128Xsl64::new(1);

    let batch_size = 32;
    let shape = [2, 8, 8];
    let mut sgd = NN2Regression::new(shape, batch_size, SGD::default(), SGD::default());
    let mut momentum = NN2Regression::new(
        shape,
        batch_size,
        MomentumSGD::default(),
        MomentumSGD::default(),
    );
    let mut ada_delta =
        NN2Regression::new(shape, batch_size, AdaDelta::default(), AdaDelta::default());
    let mut adam = NN2Regression::new(shape, batch_size, Adam::default(), Adam::default());

    println!("# epoch sgd momentum ada_delta adam");
    for epoch in 1..=10_000 {
        let (x, t) = gen_case(batch_size, &mut random);
        println!(
            "{} {} {} {} {}",
            epoch,
            sgd.train(&x, &t),
            momentum.train(&x, &t),
            ada_delta.train(&x, &t),
            adam.train(&x, &t),
        );
    }
}
