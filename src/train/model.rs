use std::io::{Read, Write};

use ndarray::{Array2, Ix1, Ix2, Zip};
use rand::{Rng, SeedableRng};
use rand_pcg::Mcg128Xsl64;

use super::{layer::*, optimizer::Optimizer};
use crate::Float;

type DenseNN1<Ow, Ob> = Layers<Dense<Ow, Ob>, ReLU>;
type DenseNN2<Ow, Ob> = Layers<Layers<DenseNN1<Ow, Ob>, Dense<Ow, Ob>>, ReLU>;
type DenseNN3<Ow, Ob> = Layers<Layers<DenseNN2<Ow, Ob>, Dense<Ow, Ob>>, ReLU>;
type DenseNN4<Ow, Ob> = Layers<Layers<DenseNN3<Ow, Ob>, Dense<Ow, Ob>>, ReLU>;
type DenseNN5<Ow, Ob> = Layers<Layers<DenseNN4<Ow, Ob>, Dense<Ow, Ob>>, ReLU>;
type DenseNN6<Ow, Ob> = Layers<Layers<DenseNN5<Ow, Ob>, Dense<Ow, Ob>>, ReLU>;

fn he_sig(n: usize) -> Float {
    (2.0 / n as Float).sqrt()
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

#[derive(Debug, Clone, PartialEq)]
pub struct NN5Regression<Ow, Ob>
where
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    nn: Layers<DenseNN5<Ow, Ob>, Dense<Ow, Ob>>,
    input: Array2<Float>,
    output: Array2<Float>,
    grad: Array2<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NN6Regression<Ow, Ob>
where
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    nn: Layers<DenseNN6<Ow, Ob>, Dense<Ow, Ob>>,
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
clone_build_nn!(build_nn5, DenseNN5, build_nn4, 5);
clone_build_nn!(build_nn6, DenseNN6, build_nn5, 6);

fn decode_nn1<R, Ow, Ob>(
    reader: &mut R,
    batch_size: usize,
    opt_w: Ow,
    opt_b: Ob,
) -> DenseNN1<Ow, Ob>
where
    R: Read,
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    use crate::predict::{Dense as PredictDense, Layer as PredictLayer};
    let dense = PredictDense::decode(reader).into_train(batch_size, opt_w, opt_b);
    let relu = ReLU::new(dense.output_size().unwrap(), batch_size);
    dense.synthesize(relu)
}

macro_rules! clone_decode_nn {
    ($func:ident, $ret:ident, $call:ident) => {
        fn $func<R, Ow, Ob>(reader: &mut R, batch_size: usize, opt_w: Ow, opt_b: Ob) -> $ret<Ow, Ob>
        where
            R: Read,
            Ow: Optimizer<Ix2> + Clone,
            Ob: Optimizer<Ix1> + Clone,
        {
            use crate::predict::{Dense as PredictDense, Layer as PredictLayer};
            let nn = $call(reader, batch_size, opt_w.clone(), opt_b.clone());
            let dense = PredictDense::decode(reader).into_train(batch_size, opt_w, opt_b);
            let relu = ReLU::new(dense.output_size().unwrap(), batch_size);
            nn.synthesize(dense).synthesize(relu)
        }
    };
}

clone_decode_nn!(decode_nn2, DenseNN2, decode_nn1);
clone_decode_nn!(decode_nn3, DenseNN3, decode_nn2);
clone_decode_nn!(decode_nn4, DenseNN4, decode_nn3);
clone_decode_nn!(decode_nn5, DenseNN5, decode_nn4);
clone_decode_nn!(decode_nn6, DenseNN6, decode_nn5);

fn decode_output<R, Ow, Ob>(
    reader: &mut R,
    batch_size: usize,
    opt_w: Ow,
    opt_b: Ob,
) -> Dense<Ow, Ob>
where
    R: Read,
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    use crate::predict::{Dense as PredictDense, Layer as PredictLayer};
    PredictDense::decode(reader).into_train(batch_size, opt_w, opt_b)
}

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
        loss / self.nn.batch_size() as Float
    }

    pub fn decode<R: Read>(
        reader: &mut R,
        batch_size: usize,
        opt_w: Ow,
        opt_b: Ob,
    ) -> NN1Regression<Ow, Ob> {
        let nn = decode_nn1(reader, batch_size, opt_w.clone(), opt_b.clone())
            .synthesize(decode_output(reader, batch_size, opt_w, opt_b));
        let input_size = nn.input_size().unwrap();
        NN1Regression {
            nn,
            input: Array2::zeros((batch_size, input_size)),
            output: Array2::zeros((batch_size, 1)),
            grad: Array2::zeros((batch_size, 1)),
        }
    }

    pub fn encode<W: Write>(&self, writer: &mut W) {
        self.nn.encode(writer);
    }
}

macro_rules! impl_nn {
    ($name:ident, $builder:ident, $decoder:ident, $n:expr) => {
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
                    input: Array2::zeros((batch_size, shape[0])),
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

            pub fn decode<R: Read>(
                reader: &mut R,
                batch_size: usize,
                opt_w: Ow,
                opt_b: Ob,
            ) -> $name<Ow, Ob> {
                let nn = $decoder(reader, batch_size, opt_w.clone(), opt_b.clone())
                    .synthesize(decode_output(reader, batch_size, opt_w, opt_b));
                let input_size = nn.input_size().unwrap();
                $name {
                    nn,
                    input: Array2::zeros((batch_size, input_size)),
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
                        let x = y - t;
                        loss += x * x * 0.5;
                        *grad = x;
                    });
                self.nn.backward(&self.grad, &mut self.input);
                self.nn.update();
                loss / self.nn.batch_size() as Float
            }

            pub fn encode<W: Write>(&self, writer: &mut W) {
                self.nn.encode(writer);
            }
        }
    };
}

impl_nn!(NN2Regression, build_nn2, decode_nn2, 2);
impl_nn!(NN3Regression, build_nn3, decode_nn3, 3);
impl_nn!(NN4Regression, build_nn4, decode_nn4, 4);
impl_nn!(NN5Regression, build_nn5, decode_nn5, 5);
impl_nn!(NN6Regression, build_nn6, decode_nn6, 6);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{Adam, SGD};
    use ndarray::arr2;

    #[test]
    fn smoke() {
        let batch_size = 1;
        let mut nn = NN1Regression::new([2, 20], batch_size, SGD::default(), Adam::default());
        nn.train(&arr2(&[[1.0, 2.0]]), &arr2(&[[3.0]]));
        let mut nn = NN2Regression::new([2, 20, 15], batch_size, Adam::default(), SGD::default());
        nn.train(&arr2(&[[1.0, 2.0]]), &arr2(&[[3.0]]));
        let mut nn =
            NN3Regression::new([2, 20, 5, 15], batch_size, SGD::default(), Adam::default());
        nn.train(&arr2(&[[1.0, 2.0]]), &arr2(&[[3.0]]));
        let mut nn = NN4Regression::new(
            [2, 4, 8, 16, 32],
            batch_size,
            Adam::default(),
            SGD::default(),
        );
        nn.train(&arr2(&[[1.0, 2.0]]), &arr2(&[[3.0]]));
    }
}
