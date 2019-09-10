use ndarray::{arr1, arr2, Array2};
use rand::Rng;
use rand_pcg::Mcg128Xsl64;

use rust_nn::Float;
use rust_nn::{
    predict::{self, Regression as PRegression},
    train::{self, Adam, Regression as TRegression, SGD},
};

fn gen_case<R: Rng>(batch_size: usize, random: &mut R) -> (Array2<Float>, Array2<Float>) {
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
}

fn main() {
    let mut random = Mcg128Xsl64::new(1);

    let batch_size = 100;
    let mut model =
        train::NN2Regression::new([2, 5, 5], batch_size, SGD::default(), SGD::default());

    for _ in 1..=100 {
        let (x, t) = gen_case(batch_size, &mut random);
        model.train(&x, &t);
    }

    // save
    let mut writer = Vec::new();
    model.encode(&mut writer);

    // load for train
    let mut model = {
        let mut reader = &writer[..];
        train::NN2Regression::decode(
            &mut reader,
            batch_size * 2,
            Adam::default(),
            Adam::default(),
        )
    };

    // Re-train
    for _ in 1..=100 {
        let (x, t) = gen_case(batch_size * 2, &mut random);
        model.train(&x, &t);
    }

    // load
    let mut model = {
        let mut reader = &writer[..];
        predict::NN2Regression::new(&mut reader)
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
}
