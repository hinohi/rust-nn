use ndarray::{arr1, arr2};
use rand::Rng;
use rand_pcg::Mcg128Xsl64;

use rust_nn::Float;
use rust_nn::{
    predict,
    train::{self, SGD},
};

fn main() {
    let mut random = Mcg128Xsl64::new(1);

    let batch_size = 100;
    let mut model = train::NN1Regression::new([2, 5], batch_size, SGD::default(), SGD::default());

    for _ in 1..=10000 {
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
        model.train(&x, &t);
    }

    // save
    let mut writer = Vec::new();
    model.encode(&mut writer);

    // load
    let mut model = {
        let mut reader = &writer[..];
        predict::NN1Regression::new(&mut reader)
    };

    // 検証
    for x in -100..=100 {
        let x = x as Float / 100.0;
        for y in -100..=100 {
            let y = y as Float / 100.0;
            let t = model.predict(&mut arr1(&[x, y]));
            println!("{} {} {}", x, y, t);
        }
        print!("\n\n");
    }

    // load for train
    let _ = {
        let mut reader = &writer[..];
        train::NN1Regression::decode(&mut reader, batch_size * 2, SGD::default(), SGD::default())
    };
}
