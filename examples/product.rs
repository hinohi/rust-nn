use ndarray::{arr1, arr2};
use rand::Rng;
use rand_pcg::Mcg128Xsl64;
use rust_nn::{
    predict::{NN2Regression as RegressionPredictor, Regression as PredictRegression},
    train::{Adam, NN2Regression as RegressionTrainer, Regression as TrainRegression},
    Float,
};

fn main() {
    // when seed=1,7,13,17... train is failed...
    let mut random = Mcg128Xsl64::new(5);

    // batch size is fixed
    let batch_size = 100;
    // build a model, that has 1 input-layer (dim=2), 2 hidden-layers (dim=4,4)
    let mut model = RegressionTrainer::with_random(
        &mut random,
        [2, 4, 4],
        batch_size,
        Adam::default(),
        Adam::default(),
    );

    // training loop
    for epoch in 1..=10_000 {
        // make data
        let (x, t) = {
            let mut x = Vec::with_capacity(batch_size);
            let mut t = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let a = random.gen_range(-2.0..=2.0);
                let b = random.gen_range(-2.0..=2.0);
                let c = a * b;
                x.push([a, b]);
                t.push([c]);
            }
            (arr2(&x), arr2(&t))
        };
        // train mini-batch and calc loss.
        // LOSS of Regression model is MSE.
        let loss = model.train(&x, &t);
        if epoch % 16 == 0 {
            eprintln!("{} {:.6}", epoch, loss);
        }
    }

    let mut predictor = {
        let mut buf = Vec::new();
        model.encode(&mut buf);
        RegressionPredictor::new(&mut buf.as_slice())
    };
    let div = 32;
    for xi in -div..=div {
        for yi in -div..=div {
            let x = 2.0 * xi as Float / div as Float;
            let y = 2.0 * yi as Float / div as Float;
            println!(
                "{} {} {} {}",
                x,
                y,
                x * y,
                predictor.predict(&arr1(&[x, y])),
            );
        }
        println!();
    }
}
