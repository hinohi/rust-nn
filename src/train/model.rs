use ndarray::{Array2, Ix1, Ix2, Zip};
use rand::{Rng, SeedableRng};
use rand_pcg::Mcg128Xsl64;

use super::{layer::*, optimizer::Optimizer};
use crate::Float;

type DenseNN1<Ow, Ob> = Layers<Dense<Ow, Ob>, ReLU>;
type DenseNN2<Ow, Ob> = Layers<Layers<DenseNN1<Ow, Ob>, Dense<Ow, Ob>>, ReLU>;
type DenseNN3<Ow, Ob> = Layers<Layers<DenseNN2<Ow, Ob>, Dense<Ow, Ob>>, ReLU>;
type DenseNN4<Ow, Ob> = Layers<Layers<DenseNN3<Ow, Ob>, Dense<Ow, Ob>>, ReLU>;

fn he_sig(n: usize) -> f64 {
    (2.0 / n as f64).sqrt()
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN1Regression<Ow, Ob>
where
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    nn: Layers<DenseNN1<Ow, Ob>, Dense<Ow, Ob>>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN2Regression<Ow, Ob>
where
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    nn: Layers<DenseNN2<Ow, Ob>, Dense<Ow, Ob>>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN3Regression<Ow, Ob>
where
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    nn: Layers<DenseNN3<Ow, Ob>, Dense<Ow, Ob>>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN4Regression<Ow, Ob>
where
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    nn: Layers<DenseNN4<Ow, Ob>, Dense<Ow, Ob>>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

fn build_nn1<R, Ow, Ob>(
    random: &mut R,
    shape: &[usize],
    batch_size: usize,
    opt_w: Ow,
    opt_b: Ob,
) -> DenseNN1<Ow, Ob>
where
    R: Rng,
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    Dense::from_normal(
        random,
        shape[0],
        shape[1],
        batch_size,
        he_sig(shape[0]),
        opt_w,
        opt_b,
    )
    .synthesize(ReLU::new(shape[1], batch_size))
}

macro_rules! clone_build_nn {
    ($func:ident, $ret:ident, $call:ident, $n:expr) => {
        fn $func<R, Ow, Ob>(
            random: &mut R,
            shape: &[usize],
            batch_size: usize,
            opt_w: Ow,
            opt_b: Ob,
        ) -> $ret<Ow, Ob>
        where
            R: Rng,
            Ow: Optimizer<Ix2> + Clone,
            Ob: Optimizer<Ix1> + Clone,
        {
            $call(random, shape, batch_size, opt_w.clone(), opt_b.clone())
                .synthesize(Dense::from_normal(
                    random,
                    shape[$n - 1],
                    shape[$n],
                    batch_size,
                    he_sig(shape[$n - 1]),
                    opt_w,
                    opt_b,
                ))
                .synthesize(ReLU::new(shape[$n], batch_size))
        }
    };
}

clone_build_nn!(build_nn2, DenseNN2, build_nn1, 2);
clone_build_nn!(build_nn3, DenseNN3, build_nn2, 3);
clone_build_nn!(build_nn4, DenseNN4, build_nn3, 4);

impl<Ow, Ob> NN1Regression<Ow, Ob>
where
    Ow: Optimizer<Ix2> + Clone,
    Ob: Optimizer<Ix1> + Clone,
{
    pub fn with_random<R>(
        random: &mut R,
        shape: [usize; 2],
        batch_size: usize,
        opt_w: Ow,
        opt_b: Ob,
    ) -> NN1Regression<Ow, Ob>
    where
        R: Rng,
    {
        let nn = build_nn1(random, &shape, batch_size, opt_w.clone(), opt_b.clone()).synthesize(
            Dense::from_normal(
                random,
                shape[1],
                1,
                batch_size,
                he_sig(shape[1]),
                opt_w,
                opt_b,
            ),
        );
        NN1Regression {
            nn,
            input: Array2::zeros((batch_size, shape[0])),
            output: Array2::zeros((batch_size, 1)),
            grad: Array2::zeros((batch_size, 1)),
        }
    }

    pub fn new(
        shape: [usize; 2],
        batch_size: usize,
        opt_w: Ow,
        opt_b: Ob,
    ) -> NN1Regression<Ow, Ob> {
        let mut random = Mcg128Xsl64::from_entropy();
        Self::with_random(&mut random, shape, batch_size, opt_w, opt_b)
    }

    pub fn train(&mut self, x: &Array2<Float>, t: &Array2<Float>) -> Float {
        self.nn.forward(x, &mut self.output);
        let mut loss = 0.0;
        Zip::from(&mut self.grad)
            .and(&self.output)
            .and(t)
            .apply(|grad, &y, &t| {
                let x = y - t;
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

macro_rules! impl_nn {
    ($name:ident, $builder:ident, $n:expr) => {
        impl<Ow, Ob> $name<Ow, Ob>
        where
            Ow: Optimizer<Ix2> + Clone,
            Ob: Optimizer<Ix1> + Clone,
        {
            pub fn with_random<R>(
                random: &mut R,
                shape: [usize; $n + 1],
                batch_size: usize,
                opt_w: Ow,
                opt_b: Ob,
            ) -> $name<Ow, Ob>
            where
                R: Rng,
            {
                let nn = $builder(random, &shape, batch_size, opt_w.clone(), opt_b.clone())
                    .synthesize(Dense::from_normal(
                        random,
                        shape[$n],
                        1,
                        batch_size,
                        he_sig(shape[$n]),
                        opt_w,
                        opt_b,
                    ));
                $name {
                    nn,
                    input: Array2::zeros((batch_size, shape[$n - 1])),
                    output: Array2::zeros((batch_size, 1)),
                    grad: Array2::zeros((batch_size, 1)),
                }
            }

            pub fn new(
                shape: [usize; $n + 1],
                batch_size: usize,
                opt_w: Ow,
                opt_b: Ob,
            ) -> $name<Ow, Ob> {
                let mut random = Mcg128Xsl64::from_entropy();
                Self::with_random(&mut random, shape, batch_size, opt_w, opt_b)
            }

            pub fn train(&mut self, x: &Array2<Float>, t: &Array2<Float>) -> Float {
                self.nn.forward(x, &mut self.output);
                let mut loss = 0.0;
                Zip::from(&mut self.grad)
                    .and(&self.output)
                    .and(t)
                    .apply(|grad, &y, &t| {
                        let x = y - t;
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

impl_nn!(NN2Regression, build_nn2, 2);
impl_nn!(NN3Regression, build_nn3, 3);
impl_nn!(NN4Regression, build_nn4, 4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::SGD;

    #[test]
    fn smoke() {
        let batch_size = 5;
        let _ = NN1Regression::new(
            [10, 20],
            batch_size,
            SGD::new(1e-3, batch_size),
            SGD::new(1e-3, batch_size),
        );
        let _ = NN2Regression::new(
            [10, 20, 15],
            batch_size,
            SGD::new(1e-3, batch_size),
            SGD::new(1e-3, batch_size),
        );
        let _ = NN3Regression::new(
            [10, 20, 5, 15],
            batch_size,
            SGD::new(1e-3, batch_size),
            SGD::new(1e-3, batch_size),
        );
        let _ = NN4Regression::new(
            [2, 4, 8, 16, 32],
            batch_size,
            SGD::new(1e-3, batch_size),
            SGD::new(1e-3, batch_size),
        );
    }
}
