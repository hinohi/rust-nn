use ndarray::{arr2, Array2};
use rand::Rng;
use rand_pcg::Mcg128Xsl64;
use rust_nn::train::*;

fn main() {
    let mut random = Mcg128Xsl64::new(1);

    let batch_size = 100;
    let hid1_size = 5;
    let hid2_size = 5;
    let mut model = Dense::from_normal(&mut random, 2, hid1_size, batch_size, 1.0)
        .with_learning_rate(1e-2)
        .synthesize(ReLU::new(hid1_size, batch_size))
        .synthesize(
            Dense::from_normal(&mut random, hid1_size, hid2_size, batch_size, 0.5)
                .with_learning_rate(1e-2),
        )
        .synthesize(ReLU::new(hid1_size, batch_size))
        .synthesize(
            Dense::from_normal(&mut random, hid2_size, 1, batch_size, 0.5).with_learning_rate(1e-2),
        );

    for epoch in 1..=10000 {
        // make data
        let (input, ans) = {
            let mut input = Vec::with_capacity(batch_size);
            let mut ans = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let a = random.gen_range(-1.0, 1.0);
                let b = random.gen_range(-1.0, 1.0);
                let c = if 0.0 < a * b { 1.0 } else { -1.0 };
                input.push([a, b]);
                ans.push(c);
            }
            (arr2(&input), ans)
        };

        // forward
        let mut output = Array2::zeros((batch_size, 1));
        model.forward(&input, &mut output);

        // calc loss and loss-grad
        let (loss, grad) = {
            let mut loss = 0.0;
            let mut grad = Vec::with_capacity(batch_size);
            for (&ans, output) in ans.iter().zip(output.genrows()) {
                let x = ans - output[0];
                loss += x * x * 0.5;
                grad.push([x]);
            }
            (loss / batch_size as f64, arr2(&grad))
        };

        model.backward(&grad, &mut Array2::zeros((batch_size, 2)));
        model.update();

        println!("{} {}", epoch, loss);
    }
}
