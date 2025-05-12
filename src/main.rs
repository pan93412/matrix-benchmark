use rand::Rng;
use rayon::prelude::*;
use std::fmt;
use std::time::Instant;

const SIZE: usize = 500; // Use 500 for demonstration; increase for real benchmarking

#[derive(Clone)]
struct Matrix<const ROWS: usize, const COLS: usize>([[f64; COLS]; ROWS]);

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    fn random() -> Self {
        let mut rng = rand::rng();

        // write random values to the matrix
        let data = std::array::from_fn(|_| {
            std::array::from_fn(|_| rng.random::<f64>() * 10.0)
        });

        Matrix(data)
    }

    fn zeros() -> Self {
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

                write!(f, "{val:.2}")?;
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
fn multiply<const ROWS: usize, const COLS: usize, const INNER: usize>(a: &Matrix<INNER, COLS>, b: &Matrix<ROWS, INNER>) -> Matrix<ROWS, INNER>
{
    let mut result = Matrix::zeros();

    for i in 0..ROWS {
        for j in 0..COLS {
            let mut sum = 0.0;
            {
                for k in 0..INNER {
                    sum += a.0[i][k] * b.0[k][j];
                }
            }
            result.0[i][j] = sum;
        }
    }
    result
}

// Multi-threaded matrix multiplication using Rayon
fn multiply_parallel<const ROWS: usize, const COLS: usize, const INNER: usize>(a: &Matrix<INNER, COLS>, b: &Matrix<ROWS, INNER>, num_threads: usize) -> Matrix<ROWS, INNER> {
    let mut result = Matrix::zeros();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok();

    result.0.par_iter_mut().enumerate().for_each(|(i, row)| {
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
    result
}

fn main() {
    println!("Generating random matrices...");
    let matrix_a = Matrix::<SIZE, SIZE>::random();
    let matrix_b = Matrix::<SIZE, SIZE>::random();

    println!("\n=== Matrix Multiplication Performance Test ===");
    println!("Matrix size: {SIZE}x{SIZE}");
    println!("Number of available CPU cores: {}", num_cpus::get());

    println!(
        "\n+-------------+----------------+-----------------------+-----------------------+-----------------------+---------------+"
    );
    println!(
        "|    Skill    |  No. threads   |    1-round (ms)       |     2-round (ms)      |     3-round (ms)      |  Average (ms) |"
    );
    println!(
        "+-------------+----------------+-----------------------+-----------------------+-----------------------+---------------+"
    );

    // [A] For-loops (1 thread)
    let forloop_times = (0..3).map(|_| {
        let start = Instant::now();
        let _ = multiply(&matrix_a, &matrix_b);
        start.elapsed().as_secs_f64() * 1000.0
    }).collect::<Vec<_>>();
    let forloop_avg = forloop_times.iter().sum::<f64>() / 3.0;

    // [B1] Multithread (50 threads)
    let b1_times = (0..3).map(|_| {
        let start = Instant::now();
        let _ = multiply_parallel(&matrix_a, &matrix_b, 50);
        start.elapsed().as_secs_f64() * 1000.0
    }).collect::<Vec<_>>();
    let b1_avg = b1_times.iter().sum::<f64>() / 3.0;

    // [B2] Multithread (10 threads)
    let b2_times = (0..3).map(|_| {
        let start = Instant::now();
        let _ = multiply_parallel(&matrix_a, &matrix_b, 10);
        start.elapsed().as_secs_f64() * 1000.0
    }).collect::<Vec<_>>();
    let b2_avg = b2_times.iter().sum::<f64>() / 3.0;

    println!(
        "| [A]         | 1              | {:21.2} | {:21.2} | {:21.2} | {:13.2} |",
        forloop_times[0], forloop_times[1], forloop_times[2], forloop_avg
    );
    println!(
        "| For-loops   | (50*50/thread) |                       |                       |                       |               |"
    );

    println!(
        "| [B1]        | 50             | {:21.2} | {:21.2} | {:21.2} | {:13.2} |",
        b1_times[0], b1_times[1], b1_times[2], b1_avg
    );
    println!(
        "| Multithread | (50*1/thread)  |                       |                       |                       |               |"
    );

    println!(
        "| [B2]        | 10             | {:21.2} | {:21.2} | {:21.2} | {:13.2} |",
        b2_times[0], b2_times[1], b2_times[2], b2_avg
    );
    println!(
        "| Multithread | (50*5/thread)  |                       |                       |                       |               |"
    );

    println!(
        "| Differences | 49             |                       |                       |                       | {:13.2} |",
        forloop_avg - b1_avg
    );
    println!(
        "| [B1 - A]    |                |                       |                       |                       |               |"
    );

    println!(
        "| Differences | 9              |                       |                       |                       | {:13.2} |",
        forloop_avg - b2_avg
    );
    println!(
        "| [B2 - A]    |                |                       |                       |                       |               |"
    );

    println!(
        "+-------------+----------------+-----------------------+-----------------------+-----------------------+---------------+"
    );
}
