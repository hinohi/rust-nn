# Neural Network Framework for Rust

## core features

* No overhead in prediction, because of separating prediction and learning.
* Strong typed Layers.

## Example

Train "product of two numbers".

```rust
use ndarray::arr2;
use rand::Rng;
use rand_pcg::Mcg128Xsl64;
use rust_nn::train::*;

fn main() {
    let mut random = Mcg128Xsl64::new(1);

    // batch size is fixed
    let batch_size = 100;
    // build a model, that has 1 input-layer (dim=2), 4 hidden-layers (dim=8,16,16,8)
    let mut model = NN3Regression::new(
        [2, 8, 16, 16, 8],
        batch_size,
        Adam::default(),
        Adam::default(),
    );

    // training loop
    for epoch in 1..=10000 {
        // make data
        let (x, t) = {
            let mut x = Vec::with_capacity(batch_size);
            let mut t = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let a = random.gen_range(-2.0, 2.0);
                let b = random.gen_range(-2.0, 2.0);
                let c = a * b;
                x.push([a, b]);
                t.push([c]);
            }
            (arr2(&x), arr2(&t))
        };
        // train mini-batch and calc loss.
        // LOSS of Regression model is MSE.
        let loss = model.train(&x, &t);
        println!("{} {}", epoch, loss);
    }
}

```

## TODO

* [ ] impl classify.
* [ ] Add document.
