use ndarray::{Array2, Zip};
use rand::{Rng, SeedableRng};
use rand_pcg::Mcg128Xsl64;

use super::layer::*;

type DenseNN1 = Layers<Dense, ReLU>;
type DenseNN2 = Layers<Layers<DenseNN1, Dense>, ReLU>;
type DenseNN3 = Layers<Layers<DenseNN2, Dense>, ReLU>;

fn he_sig(n: usize) -> f64 {
    (2.0 / n as f64).sqrt()
}

fn build_nn1<R: Rng>(
    random: &mut R,
    shape: &[usize],
    batch_size: usize,
    learning_rate: Float,
) -> Layers<DenseNN1, Dense> {
    assert_eq!(shape.len(), 2);
    Dense::from_normal(random, shape[0], shape[1], batch_size, he_sig(shape[0]))
        .with_learning_rate(learning_rate)
        .synthesize(ReLU::new(shape[1], batch_size))
        .synthesize(
            Dense::from_normal(random, shape[1], 1, batch_size, he_sig(shape[1]))
                .with_learning_rate(learning_rate),
        )
}

fn build_nn2<R: Rng>(
    random: &mut R,
    shape: &[usize],
    batch_size: usize,
    learning_rate: Float,
) -> Layers<DenseNN2, Dense> {
    assert_eq!(shape.len(), 3);
    Dense::from_normal(random, shape[0], shape[1], batch_size, he_sig(shape[0]))
        .with_learning_rate(learning_rate)
        .synthesize(ReLU::new(shape[1], batch_size))
        .synthesize(
            Dense::from_normal(random, shape[1], shape[2], batch_size, he_sig(shape[1]))
                .with_learning_rate(learning_rate),
        )
        .synthesize(ReLU::new(shape[2], batch_size))
        .synthesize(
            Dense::from_normal(random, shape[2], 1, batch_size, he_sig(shape[1]))
                .with_learning_rate(learning_rate),
        )
}

fn build_nn3<R: Rng>(
    random: &mut R,
    shape: &[usize],
    batch_size: usize,
    learning_rate: Float,
) -> Layers<DenseNN3, Dense> {
    assert_eq!(shape.len(), 4);
    Dense::from_normal(random, shape[0], shape[1], batch_size, he_sig(shape[0]))
        .with_learning_rate(learning_rate)
        .synthesize(ReLU::new(shape[1], batch_size))
        .synthesize(
            Dense::from_normal(random, shape[1], shape[2], batch_size, he_sig(shape[1]))
                .with_learning_rate(learning_rate),
        )
        .synthesize(ReLU::new(shape[2], batch_size))
        .synthesize(
            Dense::from_normal(random, shape[2], shape[3], batch_size, he_sig(shape[1]))
                .with_learning_rate(learning_rate),
        )
        .synthesize(ReLU::new(shape[3], batch_size))
        .synthesize(
            Dense::from_normal(random, shape[3], 1, batch_size, he_sig(shape[1]))
                .with_learning_rate(learning_rate),
        )
}

pub struct NNRegression {
    nn: Box<dyn Layer>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

impl NNRegression {
    pub fn new(shape: &[usize], batch_size: usize, learning_rate: Float) -> NNRegression {
        let mut random = Mcg128Xsl64::from_entropy();
        Self::with_random(&mut random, shape, batch_size, learning_rate)
    }

    pub fn with_random<R: Rng>(
        random: &mut R,
        shape: &[usize],
        batch_size: usize,
        learning_rate: Float,
    ) -> NNRegression {
        let nn: Box<dyn Layer> = match shape.len() {
            2 => Box::new(build_nn1(random, shape, batch_size, learning_rate)),
            3 => Box::new(build_nn2(random, shape, batch_size, learning_rate)),
            4 => Box::new(build_nn3(random, shape, batch_size, learning_rate)),
            _ => panic!("Unsupported NN: {:?}", shape),
        };
        NNRegression {
            nn,
            input: Array2::zeros((batch_size, shape[0])),
            output: Array2::zeros((batch_size, 1)),
            grad: Array2::zeros((batch_size, 1)),
        }
    }

    pub fn train(&mut self, x: &Array2<Float>, t: &Array2<Float>) -> Float {
        self.nn.forward(x, &mut self.output);
        let mut loss = 0.0;
        Zip::from(&mut self.grad)
            .and(&self.output)
            .and(t)
            .apply(|grad, &y, &t| {
                let x = t - y;
                loss += x * x * 0.5;
                *grad = x;
            });
        self.nn.backward(&self.grad, &mut self.input);
        self.nn.update();
        loss / self.nn.batch_size().unwrap() as Float
    }
}
