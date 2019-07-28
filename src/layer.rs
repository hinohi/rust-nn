use ndarray::{Array1, Array2, Zip};

type Float = f64;

pub trait Layer {
    type Input;
    type Output;

    fn forward(&self, input: &Self::Input, output: &mut Self::Output);
}

pub struct Dense {
    w: Array2<Float>,
    b: Array1<Float>,
}

impl Layer for Dense {
    type Input = Array1<Float>;
    type Output = Array1<Float>;

    fn forward(&self, input: &Self::Input, output: &mut Self::Output) {
        Zip::from(output)
            .and(self.w.genrows())
            .and(&self.b)
            .apply(|y, w, &b| {
                *y = w.dot(input) + b;
            });
    }
}

pub struct ReLU;

impl Layer for ReLU {
    type Input = Array1<Float>;
    type Output = Array1<Float>;

    fn forward(&self, input: &Self::Input, output: &mut Self::Output) {
        Zip::from(output).and(input).apply(|y, &x| {
            if 0.0 < x {
                *y = x;
            } else {
                *y = 0.0;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn smoke_dense() {
        let dense = Dense {
            w: arr2(&[[1.0, 2.0], [3.0, -4.0]]),
            b: arr1(&[2.0, -3.0]),
        };
        let input = arr1(&[0.5, -1.5]);
        let mut output = arr1(&[0.0, 0.0]);
        dense.forward(&input, &mut output);
        assert_eq!(output, arr1(&[0.5 - 3.0 + 2.0, 1.5 + 6.0 - 3.0]));
    }

    #[test]
    fn smoke_relu() {
        let r = ReLU {};
        let input = arr1(&[0.5, -1.5, 0.0, 2.0]);
        let mut output = arr1(&[1.0; 4]);
        r.forward(&input, &mut output);
        assert_eq!(output, arr1(&[0.5, 0.0, 0.0, 2.0]));
    }
}
