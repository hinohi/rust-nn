use ndarray::{Array2, Zip};
use rand::{Rng, SeedableRng};
use rand_pcg::Mcg128Xsl64;

use super::layer::*;
use crate::Float;

type DenseNN1 = Layers<Dense, ReLU>;
type DenseNN2 = Layers<Layers<DenseNN1, Dense>, ReLU>;
type DenseNN3 = Layers<Layers<DenseNN2, Dense>, ReLU>;
type DenseNN4 = Layers<Layers<DenseNN3, Dense>, ReLU>;

fn he_sig(n: usize) -> f64 {
    (2.0 / n as f64).sqrt()
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN1Regression {
    nn: Layers<DenseNN1, Dense>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN2Regression {
    nn: Layers<DenseNN2, Dense>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN3Regression {
    nn: Layers<DenseNN3, Dense>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN4Regression {
    nn: Layers<DenseNN4, Dense>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

impl NN1Regression {
    pub fn with_random<R: Rng>(
        random: &mut R,
        shape: [usize; 2],
        batch_size: usize,
        learning_rate: Float,
    ) -> NN1Regression {
        let nn = Dense::from_normal(random, shape[0], shape[1], batch_size, he_sig(shape[0]))
            .with_learning_rate(learning_rate)
            .synthesize(ReLU::new(shape[1], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[1], 1, batch_size, he_sig(shape[1]))
                    .with_learning_rate(learning_rate),
            );
        NN1Regression {
            nn,
            input: Array2::zeros((batch_size, shape[0])),
            output: Array2::zeros((batch_size, 1)),
            grad: Array2::zeros((batch_size, 1)),
        }
    }

    pub fn new(shape: [usize; 2], batch_size: usize, learning_rate: Float) -> NN1Regression {
        let mut random = Mcg128Xsl64::from_entropy();
        Self::with_random(&mut random, shape, batch_size, learning_rate)
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

    pub fn get_inner(&self) -> &impl Layer {
        &self.nn
    }
}

impl NN2Regression {
    pub fn with_random<R: Rng>(
        random: &mut R,
        shape: [usize; 3],
        batch_size: usize,
        learning_rate: Float,
    ) -> NN2Regression {
        let nn = Dense::from_normal(random, shape[0], shape[1], batch_size, he_sig(shape[0]))
            .with_learning_rate(learning_rate)
            .synthesize(ReLU::new(shape[1], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[1], shape[2], batch_size, he_sig(shape[1]))
                    .with_learning_rate(learning_rate),
            )
            .synthesize(ReLU::new(shape[2], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[2], 1, batch_size, he_sig(shape[2]))
                    .with_learning_rate(learning_rate),
            );
        NN2Regression {
            nn,
            input: Array2::zeros((batch_size, shape[0])),
            output: Array2::zeros((batch_size, 1)),
            grad: Array2::zeros((batch_size, 1)),
        }
    }
}

impl NN3Regression {
    pub fn with_random<R: Rng>(
        random: &mut R,
        shape: [usize; 4],
        batch_size: usize,
        learning_rate: Float,
    ) -> NN3Regression {
        let nn = Dense::from_normal(random, shape[0], shape[1], batch_size, he_sig(shape[0]))
            .with_learning_rate(learning_rate)
            .synthesize(ReLU::new(shape[1], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[1], shape[2], batch_size, he_sig(shape[1]))
                    .with_learning_rate(learning_rate),
            )
            .synthesize(ReLU::new(shape[2], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[2], shape[3], batch_size, he_sig(shape[2]))
                    .with_learning_rate(learning_rate),
            )
            .synthesize(ReLU::new(shape[3], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[3], 1, batch_size, he_sig(shape[3]))
                    .with_learning_rate(learning_rate),
            );
        NN3Regression {
            nn,
            input: Array2::zeros((batch_size, shape[0])),
            output: Array2::zeros((batch_size, 1)),
            grad: Array2::zeros((batch_size, 1)),
        }
    }
}

impl NN4Regression {
    pub fn with_random<R: Rng>(
        random: &mut R,
        shape: [usize; 5],
        batch_size: usize,
        learning_rate: Float,
    ) -> NN4Regression {
        let nn = Dense::from_normal(random, shape[0], shape[1], batch_size, he_sig(shape[0]))
            .with_learning_rate(learning_rate)
            .synthesize(ReLU::new(shape[1], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[1], shape[2], batch_size, he_sig(shape[1]))
                    .with_learning_rate(learning_rate),
            )
            .synthesize(ReLU::new(shape[2], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[2], shape[3], batch_size, he_sig(shape[2]))
                    .with_learning_rate(learning_rate),
            )
            .synthesize(ReLU::new(shape[3], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[3], shape[4], batch_size, he_sig(shape[3]))
                    .with_learning_rate(learning_rate),
            )
            .synthesize(ReLU::new(shape[4], batch_size))
            .synthesize(
                Dense::from_normal(random, shape[4], 1, batch_size, he_sig(shape[4]))
                    .with_learning_rate(learning_rate),
            );
        NN4Regression {
            nn,
            input: Array2::zeros((batch_size, shape[0])),
            output: Array2::zeros((batch_size, 1)),
            grad: Array2::zeros((batch_size, 1)),
        }
    }
}

macro_rules! impl_train {
    ($cls:tt, $hid:expr) => {
        impl $cls {
            pub fn new(shape: [usize; $hid], batch_size: usize, learning_rate: Float) -> $cls {
                let mut random = Mcg128Xsl64::from_entropy();
                Self::with_random(&mut random, shape, batch_size, learning_rate)
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

            pub fn get_inner(&self) -> &impl Layer {
                &self.nn
            }
        }
    };
}

impl_train!(NN2Regression, 3);
impl_train!(NN3Regression, 4);
impl_train!(NN4Regression, 5);
