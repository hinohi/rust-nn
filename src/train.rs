use ndarray::{Array1, Array2, Zip};
use rand::Rng;
use rand_distr::{Distribution, Normal};

type Float = f64;

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
        shape: (usize, usize),
        batch: usize,
        sig: Float,
    ) -> Dense {
        let normal = Normal::new(0.0, sig).unwrap();
        Dense {
            w: Array2::from_shape_fn(shape, |_| normal.sample(random)),
            b: Array1::zeros(shape.1),
            input: Array2::zeros((batch, shape.0)),
            grad_w: Array2::zeros(shape),
            grad_b: Array1::zeros(shape.1),
            learning_rate: 0.01,
        }
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
        let r = self.learning_rate / self.input.shape()[0] as Float;
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
}
