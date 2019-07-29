use ndarray::{arr1, arr2};
use rand::Rng;
use rand_pcg::Mcg128Xsl64;

use rust_nn::predict::{Dense, Layer as PLayer, ReLU, Synthesize};
use rust_nn::train::{Layer as TLayer, NN1Regression};

fn main() {
    let mut random = Mcg128Xsl64::new(1);

    let batch_size = 100;
    let mut model = NN1Regression::new([2, 5], batch_size, 8e-3);

    for _ in 1..=100000 {
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
    model.get_inner().encode(&mut writer);
    let mut reader = &writer[..];

    // load
    let model = Dense::decode(&mut reader);
    let model = model.synthesize(ReLU::decode(&mut reader));
    let mut model = model.synthesize(Dense::decode(&mut reader));

    // 検証
    let mut output = arr1(&[0.0]);
    for x in -100..=100 {
        let x = f64::from(x) / 100.0;
        for y in -100..=100 {
            let y = f64::from(y) / 100.0;
            model.forward(&mut arr1(&[x, y]), &mut output);
            println!("{} {} {}", x, y, output[0]);
        }
        print!("\n\n");
    }
}
