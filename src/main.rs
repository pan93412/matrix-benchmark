use rand::Rng;
use rayon::prelude::*;
use std::fmt;
use std::time::Instant;

const SIZE: usize = 500; // Use 500 for demonstration; increase for real benchmarking

#[derive(Clone)]
struct Matrix(Vec<Vec<f64>>);

impl Matrix {
    fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let data = (0..rows)
            .map(|_| (0..cols).map(|_| rng.random::<f64>() * 10.0).collect())
            .collect();
        Matrix(data)
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        Matrix(vec![vec![0.0; cols]; rows])
    }

    fn rows(&self) -> usize {
        self.0.len()
    }
    fn cols(&self) -> usize {
        if self.0.is_empty() { 0 } else { self.0[0].len() }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.0.iter().enumerate() {
            write!(f, "[")?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 { write!(f, " ")?; }
                write!(f, "{:.2}", val)?;
            }
            write!(f, "]")?;
            if i < self.0.len() - 1 { writeln!(f)?; }
        }
        Ok(())
    }
}

// Single-threaded matrix multiplication (with optional SIMD)
fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let rows = a.rows();
    let cols = b.cols();
    let inner = a.cols();
    let mut result = Matrix::zeros(rows, cols);

    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            {
                for k in 0..inner {
                    sum += a.0[i][k] * b.0[k][j];
                }
            }
            result.0[i][j] = sum;
        }
    }
    result
}

// Multi-threaded matrix multiplication using Rayon
fn multiply_parallel(a: &Matrix, b: &Matrix, num_threads: usize) -> Matrix {
    let rows = a.rows();
    let cols = b.cols();
    let inner = a.cols();
    let mut result = Matrix::zeros(rows, cols);

    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().ok();

    result.0.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..cols {
            let mut sum = 0.0;
            {
                for k in 0..inner {
                    sum += a.0[i][k] * b.0[k][j];
                }
            }
            row[j] = sum;
        }
    });
    result
}

fn main() {
    println!("Generating random matrices...");
    let matrix_a = Matrix::random(SIZE, SIZE);
    let matrix_b = Matrix::random(SIZE, SIZE);

    println!("\n=== Matrix Multiplication Performance Test ===");
    println!("Matrix size: {}x{}", SIZE, SIZE);
    println!("Number of available CPU cores: {}", num_cpus::get());

    println!("\n+-------------+----------------+-----------------------+-----------------------+-----------------------+---------------+");
    println!("|    Skill    |  No. threads   |    1-round (ms)       |     2-round (ms)      |     3-round (ms)      |  Average (ms) |");
    println!("+-------------+----------------+-----------------------+-----------------------+-----------------------+---------------+");

    // [A] For-loops (1 thread)
    let mut forloop_times = [0.0; 3];
    for round in 0..3 {
        let start = Instant::now();
        let _ = multiply(&matrix_a, &matrix_b);
        forloop_times[round] = start.elapsed().as_secs_f64() * 1000.0;
    }
    let forloop_avg = (forloop_times[0] + forloop_times[1] + forloop_times[2]) / 3.0;

    // [B1] Multithread (50 threads)
    let mut b1_times = [0.0; 3];
    for round in 0..3 {
        let start = Instant::now();
        let _ = multiply_parallel(&matrix_a, &matrix_b, 50);
        b1_times[round] = start.elapsed().as_secs_f64() * 1000.0;
    }
    let b1_avg = (b1_times[0] + b1_times[1] + b1_times[2]) / 3.0;

    // [B2] Multithread (10 threads)
    let mut b2_times = [0.0; 3];
    for round in 0..3 {
        let start = Instant::now();
        let _ = multiply_parallel(&matrix_a, &matrix_b, 10);
        b2_times[round] = start.elapsed().as_secs_f64() * 1000.0;
    }
    let b2_avg = (b2_times[0] + b2_times[1] + b2_times[2]) / 3.0;

    println!("| [A]         | 1              | {:21.2} | {:21.2} | {:21.2} | {:13.2} |",
        forloop_times[0], forloop_times[1], forloop_times[2], forloop_avg);
    println!("| For-loops   | (50*50/thread) |                       |                       |                       |               |");

    println!("| [B1]        | 50             | {:21.2} | {:21.2} | {:21.2} | {:13.2} |",
        b1_times[0], b1_times[1], b1_times[2], b1_avg);
    println!("| Multithread | (50*1/thread)  |                       |                       |                       |               |");

    println!("| [B2]        | 10             | {:21.2} | {:21.2} | {:21.2} | {:13.2} |",
        b2_times[0], b2_times[1], b2_times[2], b2_avg);
    println!("| Multithread | (50*5/thread)  |                       |                       |                       |               |");

    println!("| Differences | 49             |                       |                       |                       | {:13.2} |",
        forloop_avg - b1_avg);
    println!("| [B1 - A]    |                |                       |                       |                       |               |");

    println!("| Differences | 9              |                       |                       |                       | {:13.2} |",
        forloop_avg - b2_avg);
    println!("| [B2 - A]    |                |                       |                       |                       |               |");

    println!("+-------------+----------------+-----------------------+-----------------------+-----------------------+---------------+");
}
