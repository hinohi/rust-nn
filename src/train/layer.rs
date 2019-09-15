use std::io::Write;

use ndarray::{Array1, Array2, Ix1, Ix2, Zip};
use ndarray_parallel::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::optimizer::Optimizer;
use crate::Float;

pub trait Layer {
    fn forward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>);

    fn backward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>);

    fn update(&mut self);

    fn input_size(&self) -> Option<usize> {
        None
    }

    fn output_size(&self) -> Option<usize> {
        None
    }

    fn batch_size(&self) -> usize;

    fn encode<W: Write>(&self, writer: &mut W);
}

/// レイヤを合成したレイヤ
#[derive(Debug, Clone, PartialEq)]
pub struct Layers<L1, L2>
where
    L1: Layer,
    L2: Layer,
{
    layer1: L1,
    layer2: L2,
    temporary: Array2<Float>,
}

impl<L1, L2> Layers<L1, L2>
where
    L1: Layer,
    L2: Layer,
{
    pub fn new(layer1: L1, layer2: L2) -> Layers<L1, L2> {
        let size = match (layer1.output_size(), layer2.input_size()) {
            (Some(size), None) | (None, Some(size)) => size,
            (Some(s1), Some(s2)) if s1 == s2 => s1,
            (a, b) => panic!("Mismatch: out={:?} in={:?}", a, b),
        };
        let batch = if layer1.batch_size() == layer2.batch_size() {
            layer2.batch_size()
        } else {
            panic!(
                "Mismatch: {} != {}",
                layer1.batch_size(),
                layer2.batch_size()
            );
        };
        Layers {
            layer1,
            layer2,
            temporary: Array2::zeros((batch, size)),
        }
    }
}

impl<L1, L2> Layer for Layers<L1, L2>
where
    L1: Layer,
    L2: Layer,
{
    #[inline]
    fn forward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        self.layer1.forward(input, &mut self.temporary);
        self.layer2.forward(&self.temporary, output);
    }

    #[inline]
    fn backward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        self.layer2.backward(input, &mut self.temporary);
        self.layer1.backward(&self.temporary, output);
    }

    #[inline]
    fn update(&mut self) {
        self.layer2.update();
        self.layer1.update();
    }

    fn input_size(&self) -> Option<usize> {
        self.layer1.input_size()
    }

    fn output_size(&self) -> Option<usize> {
        if let Some(size) = self.layer2.output_size() {
            Some(size)
        } else {
            self.layer1.output_size()
        }
    }

    fn batch_size(&self) -> usize {
        self.layer2.batch_size()
    }

    fn encode<W: Write>(&self, writer: &mut W) {
        self.layer1.encode(writer);
        self.layer2.encode(writer);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dense<Ow, Ob> {
    w: Array2<Float>,
    b: Array1<Float>,
    input: Array2<Float>,
    grad_w: Array2<Float>,
    grad_b: Array1<Float>,
    opt_w: Ow,
    opt_b: Ob,
}

impl<Ow, Ob> Dense<Ow, Ob>
where
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    pub fn from_normal<R: Rng>(
        random: &mut R,
        input_size: usize,
        output_size: usize,
        batch_size: usize,
        sig: Float,
        opt_w: Ow,
        opt_b: Ob,
    ) -> Dense<Ow, Ob> {
        let mut opt_w = opt_w;
        let mut opt_b = opt_b;
        opt_w.init((output_size, input_size), batch_size);
        opt_b.init(output_size, batch_size);

        let normal = Normal::new(0.0, sig).unwrap();
        Dense {
            w: Array2::from_shape_fn((output_size, input_size), |_| normal.sample(random)),
            b: Array1::zeros(output_size),
            input: Array2::zeros((batch_size, input_size)),
            grad_w: Array2::zeros((output_size, input_size)),
            grad_b: Array1::zeros(output_size),
            opt_w,
            opt_b,
        }
    }

    pub fn new(
        w: Array2<Float>,
        b: Array1<Float>,
        batch_size: usize,
        opt_w: Ow,
        opt_b: Ob,
    ) -> Self {
        let input_size = w.shape()[1];
        let output_size = w.shape()[0];
        let mut opt_w = opt_w;
        let mut opt_b = opt_b;
        opt_w.init((output_size, input_size), batch_size);
        opt_b.init(output_size, batch_size);
        Dense {
            w,
            b,
            input: Array2::zeros((batch_size, input_size)),
            grad_w: Array2::zeros((output_size, input_size)),
            grad_b: Array1::zeros(output_size),
            opt_w,
            opt_b,
        }
    }
}

impl<Ow, Ob> Layer for Dense<Ow, Ob>
where
    Ow: Optimizer<Ix2>,
    Ob: Optimizer<Ix1>,
{
    fn forward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        Zip::from(&mut self.input)
            .and(input)
            .par_apply(|x, &y| *x = y);
        Zip::from(output.genrows_mut())
            .and(input.genrows())
            .par_apply(|mut output, input| {
                Zip::from(&mut output)
                    .and(self.w.genrows())
                    .and(&self.b)
                    .apply(|y, w, &b| {
                        *y = w.dot(&input) + b;
                    });
            });
    }

    fn backward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        self.grad_w.fill(0.0);
        // ∂L/∂b = ∂L/∂y
        Zip::from(&mut self.grad_b)
            .and(input.gencolumns())
            .par_apply(|db, y| {
                *db = y.sum();
            });
        // ∂L/∂W = ∂L/∂y x^T
        Zip::from(input.genrows())
            // borrow checker...
            .and(self.input.clone().genrows())
            .apply(|y, x| {
                Zip::from(self.grad_w.genrows_mut())
                    .and(&y)
                    .apply(|mut dw, y| {
                        Zip::from(&mut dw).and(&x).apply(|dw, x| {
                            *dw += y * x;
                        })
                    })
            });
        // output
        Zip::from(output.genrows_mut())
            .and(input.genrows())
            .par_apply(|mut output, input| {
                Zip::from(&mut output)
                    .and(self.w.t().genrows())
                    .apply(|y, w| {
                        *y = w.dot(&input);
                    });
            });
    }

    fn update(&mut self) {
        self.opt_w.optimize(&mut self.w, &self.grad_w);
        self.opt_b.optimize(&mut self.b, &self.grad_b);
    }

    fn input_size(&self) -> Option<usize> {
        Some(self.w.shape()[1])
    }

    fn output_size(&self) -> Option<usize> {
        Some(self.w.shape()[0])
    }

    fn batch_size(&self) -> usize {
        self.input.shape()[0]
    }

    fn encode<W: Write>(&self, writer: &mut W) {
        rmp_serde::encode::write(writer, &self.w).unwrap();
        rmp_serde::encode::write(writer, &self.b).unwrap();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReLU {
    input: Array2<bool>,
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        Zip::from(output)
            .and(input)
            .and(&mut self.input)
            .par_apply(|y, &x, z| {
                if 0.0 < x {
                    *y = x;
                    *z = true;
                } else {
                    *y = 0.0;
                    *z = false;
                }
            });
    }

    fn backward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        Zip::from(output)
            .and(input)
            .and(&self.input)
            .par_apply(|y, &x, &z| {
                if z {
                    *y = x;
                } else {
                    *y = 0.0;
                }
            });
    }

    fn update(&mut self) {
        // do nothing
    }

    fn input_size(&self) -> Option<usize> {
        Some(self.input.shape()[1])
    }

    fn output_size(&self) -> Option<usize> {
        Some(self.input.shape()[1])
    }

    fn batch_size(&self) -> usize {
        self.input.shape()[0]
    }

    fn encode<W: Write>(&self, _: &mut W) {}
}

impl ReLU {
    pub fn new(size: usize, batch: usize) -> ReLU {
        ReLU {
            input: Array2::from_elem((batch, size), false),
        }
    }
}

/// レイヤ合成のインターフェース
pub trait Synthesize<Other> {
    type Output;
    fn synthesize(self, other: Other) -> Self::Output;
}

impl<L1, L2> Synthesize<L2> for L1
where
    L1: Layer,
    L2: Layer,
{
    type Output = Layers<L1, L2>;
    fn synthesize(self, other: L2) -> Self::Output {
        Layers::new(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::SGD;
    use ndarray::{arr1, arr2};

    #[test]
    fn smoke_dense() {
        let mut dense = Dense {
            w: arr2(&[[1.0, 2.0], [3.0, -4.0], [-2.0, 3.0]]),
            b: arr1(&[2.0, -3.0, 1.0]),
            input: Array2::zeros((4, 2)),
            grad_w: Array2::zeros((3, 2)),
            grad_b: Array1::zeros(3),
            opt_w: SGD::default(),
            opt_b: SGD::default(),
        };
        assert_eq!(dense.input_size(), Some(2));
        assert_eq!(dense.output_size(), Some(3));
        assert_eq!(dense.batch_size(), 4);
        let input = arr2(&[[0.5, -1.5], [0.0, 2.0], [1.0, 10.0], [-3.0, -4.0]]);
        let mut output = Array2::zeros((4, 3));
        dense.forward(&input, &mut output);
        assert_eq!(
            output,
            arr2(&[
                [0.5 - 3.0 + 2.0, 1.5 + 6.0 - 3.0, -1.0 - 4.5 + 1.0],
                [0.0 + 4.0 + 2.0, 0.0 - 8.0 - 3.0, 0.0 + 6.0 + 1.0],
                [1.0 + 20.0 + 2.0, 3.0 - 40.0 - 3.0, -2.0 + 30.0 + 1.0],
                [-3.0 - 8.0 + 2.0, -9.0 + 16.0 - 3.0, 6.0 - 12.0 + 1.0],
            ])
        );
        assert_eq!(dense.input, input);

        let input = arr2(&[
            [1.0, 1.0, 1.0],
            [-0.5, 2.0, 0.0],
            [2.0, -3.0, 4.0],
            [0.0, 0.5, -0.5],
        ]);
        let mut output = Array2::zeros((4, 2));
        dense.backward(&input, &mut output);
        assert_eq!(
            output,
            arr2(&[
                [1.0 + 3.0 - 2.0, 2.0 - 4.0 + 3.0],
                [-0.5 + 6.0 + 0.0, -1.0 - 8.0 + 0.0],
                [2.0 - 9.0 - 8.0, 4.0 + 12.0 + 12.0],
                [0.0 + 1.5 + 1.0, 0.0 - 2.0 - 1.5],
            ])
        );
        assert_eq!(
            dense.grad_b,
            arr1(&[
                1.0 - 0.5 + 2.0 + 0.0,
                1.0 + 2.0 - 3.0 + 0.5,
                1.0 + 0.0 + 4.0 - 0.5,
            ])
        );
        assert_eq!(
            dense.grad_w,
            arr2(&[
                [
                    1.0 * 0.5 - 0.5 * 0.0 + 2.0 * 1.0 + 0.0 * (-3.0),
                    1.0 * (-1.5) - 0.5 * 2.0 + 2.0 * 10.0 + 0.0 * (-4.0),
                ],
                [
                    1.0 * 0.5 + 2.0 * 0.0 - 3.0 * 1.0 + 0.5 * (-3.0),
                    1.0 * (-1.5) + 2.0 * 2.0 - 3.0 * 10.0 + 0.5 * (-4.0),
                ],
                [
                    1.0 * 0.5 + 0.0 * 0.0 + 4.0 * 1.0 - 0.5 * (-3.0),
                    1.0 * (-1.5) + 0.0 * 2.0 + 4.0 * 10.0 - 0.5 * (-4.0),
                ]
            ])
        );
        dense.update();
    }

    #[test]
    fn relu() {
        let mut relu = ReLU::new(2, 4);
        assert_eq!(relu.input_size(), Some(2));
        assert_eq!(relu.output_size(), Some(2));
        assert_eq!(relu.batch_size(), 4);
        let input = arr2(&[[1.0, 1.0], [2.0, 3.0], [-0.5, 0.0], [-2.0, -3.0]]);
        let mut output = Array2::zeros((4, 2));
        relu.forward(&input, &mut output);
        assert_eq!(
            output,
            arr2(&[[1.0, 1.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0]])
        );

        let input = arr2(&[[-1.0, 2.0], [5.0, 4.0], [-0.5, 3.0], [2.0, 3.0]]);
        let mut output = Array2::zeros((4, 2));
        relu.backward(&input, &mut output);
        assert_eq!(
            output,
            arr2(&[[-1.0, 2.0], [5.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
        );
    }

    #[test]
    fn dense_relu() {
        let dense = Dense {
            w: arr2(&[[1.0, 2.0], [3.0, -4.0], [-2.0, 3.0]]),
            b: arr1(&[2.0, -3.0, 1.0]),
            input: Array2::zeros((4, 2)),
            grad_w: Array2::zeros((3, 2)),
            grad_b: Array1::zeros(3),
            opt_w: SGD::default(),
            opt_b: SGD::default(),
        };
        let relu = ReLU::new(dense.output_size().unwrap(), dense.batch_size());
        let mut l = dense.synthesize(relu);
        assert_eq!(l.input_size(), Some(2));
        assert_eq!(l.output_size(), Some(3));
        assert_eq!(l.batch_size(), 4);

        let input = arr2(&[[0.5, -1.5], [0.0, 2.0], [1.0, 10.0], [-3.0, -4.0]]);
        let mut output = Array2::zeros((4, 3));
        l.forward(&input, &mut output);
        assert_eq!(
            output,
            arr2(&[
                [0.0, 1.5 + 6.0 - 3.0, 0.0],
                [0.0 + 4.0 + 2.0, 0.0, 0.0 + 6.0 + 1.0],
                [1.0 + 20.0 + 2.0, 0.0, -2.0 + 30.0 + 1.0],
                [0.0, -9.0 + 16.0 - 3.0, 0.0],
            ])
        );

        let input = arr2(&[
            [1.0, 1.0, 1.0],
            [-0.5, 2.0, 0.0],
            [2.0, -3.0, 4.0],
            [0.0, 0.5, -0.5],
        ]);
        let mut output = Array2::zeros((4, 2));
        // smoke test
        l.backward(&input, &mut output);
        l.update();
    }
}
