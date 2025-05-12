use std::fmt;

use rand::Rng as _;
use rayon::iter::{IndexedParallelIterator as _, IntoParallelRefMutIterator as _, ParallelIterator as _};

#[derive(Clone)]
pub struct Matrix<const ROWS: usize, const COLS: usize>([[f64; COLS]; ROWS]);

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    pub fn random() -> Self {
        let mut rng = rand::rng();

        // write random values to the matrix
        let data = std::array::from_fn(|_| {
            std::array::from_fn(|_| rng.random::<f64>() * 10.0)
        });

        Matrix(data)
    }

    pub const fn zeros() -> Self {
        Matrix([[0.0; COLS]; ROWS])
    }
}

impl<const ROWS: usize, const COLS: usize> fmt::Display for Matrix<ROWS, COLS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.0.iter().enumerate() {
            write!(f, "[")?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, " ")?;
                }

                write!(f, "{val:.6}")?;
            }
            write!(f, "]")?;
            if i < self.0.len() - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

// Single-threaded matrix multiplication
pub fn multiply<const ROWS: usize, const COLS: usize, const INNER: usize>(
    a: &Matrix<INNER, COLS>,
    b: &Matrix<ROWS, INNER>,
    target: &mut Matrix<ROWS, INNER>,
) {
    for i in 0..ROWS {
        for j in 0..COLS {
            let mut sum = 0.0;
            for k in 0..INNER {
                sum += a.0[i][k] * b.0[k][j];
            }
            target.0[i][j] = sum;
        }
    }
}

// Multi-threaded matrix multiplication using Rayon
pub fn multiply_parallel<const ROWS: usize, const COLS: usize, const INNER: usize>(a: &Matrix<INNER, COLS>, b: &Matrix<ROWS, INNER>, num_threads: usize, target: &mut Matrix<ROWS, INNER>) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok();

    target.0.par_iter_mut().enumerate().for_each(|(i, row)| {
        for (j, elem) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            {
                for k in 0..INNER {
                    sum += a.0[i][k] * b.0[k][j];
                }
            }
            *elem = sum;
        }
    });
}
