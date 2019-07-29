use ndarray::{Array1, Array2, Zip};
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub type Float = f64;

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

    fn batch_size(&self) -> Option<usize> {
        None
    }
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
            (a, b) => panic!("missmatch: out={:?} in={:?}", a, b),
        };
        let batch = match (layer1.batch_size(), layer2.batch_size()) {
            (Some(size), None) | (None, Some(size)) => size,
            (Some(s1), Some(s2)) if s1 == s2 => s1,
            _ => panic!(),
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
    fn forward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        self.layer1.forward(input, &mut self.temporary);
        self.layer2.forward(&self.temporary, output);
    }

    fn backward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        self.layer2.backward(input, &mut self.temporary);
        self.layer1.backward(&self.temporary, output);
    }

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

    fn batch_size(&self) -> Option<usize> {
        if let Some(size) = self.layer2.batch_size() {
            Some(size)
        } else {
            self.layer1.batch_size()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    w: Array2<Float>,
    b: Array1<Float>,
    input: Array2<Float>,
    grad_w: Array2<Float>,
    grad_b: Array1<Float>,
    learning_rate: Float,
}

impl Dense {
    pub fn from_normal<R: Rng>(
        random: &mut R,
        input_size: usize,
        output_size: usize,
        batch_size: usize,
        sig: Float,
    ) -> Dense {
        let normal = Normal::new(0.0, sig).unwrap();
        Dense {
            w: Array2::from_shape_fn((output_size, input_size), |_| normal.sample(random)),
            b: Array1::zeros(output_size),
            input: Array2::zeros((batch_size, input_size)),
            grad_w: Array2::zeros((output_size, input_size)),
            grad_b: Array1::zeros(output_size),
            learning_rate: 1.0 / 128.0,
        }
    }

    pub fn with_learning_rate(mut self, learning_rate: Float) -> Dense {
        self.learning_rate = learning_rate;
        self
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        Zip::from(&mut self.input).and(input).apply(|x, &y| *x = y);
        Zip::from(output.genrows_mut())
            .and(input.genrows())
            .apply(|mut output, input| {
                Zip::from(&mut output)
                    .and(self.w.genrows())
                    .and(&self.b)
                    .apply(|y, w, &b| {
                        *y = w.dot(&input) + b;
                    });
            });
    }

    fn backward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        // Zero fill (Any more good way?)
        Zip::from(&mut self.grad_w).apply(|w| *w = 0.0);
        Zip::from(&mut self.grad_b).apply(|b| *b = 0.0);
        // ∂L/∂b = ∂L/∂y
        Zip::from(input.genrows()).apply(|input| {
            Zip::from(&mut self.grad_b).and(&input).apply(|db, &y| {
                *db += y;
            });
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
            .apply(|mut output, input| {
                Zip::from(&mut output)
                    .and(self.w.t().genrows())
                    .apply(|y, w| {
                        *y = w.dot(&input);
                    });
            });
    }

    fn update(&mut self) {
        let r = self.learning_rate / self.batch_size().unwrap() as Float;
        Zip::from(&mut self.w)
            .and(&self.grad_w)
            .apply(|x, &d| *x += d * r);
        Zip::from(&mut self.b)
            .and(&self.grad_b)
            .apply(|x, &d| *x += d * r);
    }

    fn input_size(&self) -> Option<usize> {
        Some(self.w.shape()[1])
    }

    fn output_size(&self) -> Option<usize> {
        Some(self.w.shape()[0])
    }

    fn batch_size(&self) -> Option<usize> {
        Some(self.input.shape()[0])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReLU {
    input: Array2<Float>,
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        Zip::from(&mut self.input).and(input).apply(|y, &x| *y = x);
        Zip::from(output).and(input).apply(|y, &x| {
            if 0.0 < x {
                *y = x;
            } else {
                *y = 0.0;
            }
        });
    }

    fn backward(&mut self, input: &Array2<Float>, output: &mut Array2<Float>) {
        Zip::from(output)
            .and(input)
            .and(&self.input)
            .apply(|y, &x, &z| {
                if 0.0 < z {
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

    fn batch_size(&self) -> Option<usize> {
        Some(self.input.shape()[0])
    }
}

impl ReLU {
    pub fn new(size: usize, batch: usize) -> ReLU {
        ReLU {
            input: Array2::zeros((batch, size)),
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
    use ndarray::{arr1, arr2};

    #[test]
    fn smoke_dense() {
        let mut dense = Dense {
            w: arr2(&[[1.0, 2.0], [3.0, -4.0], [-2.0, 3.0]]),
            b: arr1(&[2.0, -3.0, 1.0]),
            input: Array2::zeros((4, 2)),
            grad_w: Array2::zeros((3, 2)),
            grad_b: Array1::zeros(3),
            learning_rate: 0.01,
        };
        assert_eq!(dense.input_size(), Some(2));
        assert_eq!(dense.output_size(), Some(3));
        assert_eq!(dense.batch_size(), Some(4));
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
        assert_eq!(relu.batch_size(), Some(4));
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
            learning_rate: 0.01,
        };
        let relu = ReLU::new(dense.output_size().unwrap(), dense.batch_size().unwrap());
        let mut l = dense.synthesize(relu);
        assert_eq!(l.input_size(), Some(2));
        assert_eq!(l.output_size(), Some(3));
        assert_eq!(l.batch_size(), Some(4));

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