use ndarray::arr2;
use rand::Rng;
use rand_pcg::Mcg128Xsl64;
use rust_nn::train::*;

fn main() {
    let mut random = Mcg128Xsl64::new(1);

    let batch_size = 100;
    let mut model = NN1Regression::new([2, 5], batch_size, 8e-3);

    for epoch in 1..=100000 {
        // make data
        let (x, t) = {
            let mut x = Vec::with_capacity(batch_size);
            let mut t = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let a = random.gen_range(-1.0, 1.0);
                let b = random.gen_range(-1.0, 1.0);
                let c = if 0.0 < a * b { 1.0 } else { -1.0 };
                x.push([a, b]);
                t.push([c]);
            }
            (arr2(&x), arr2(&t))
        };
        let loss = model.train(&x, &t);
        println!("{} {}", epoch, loss);
    }
}
