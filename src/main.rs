use std::time::Instant;

use matrixcc::{multiply, multiply_parallel, Matrix};

const SIZE: usize = 700; // Use 700 for demonstration; increase for real benchmarking

fn main() {
    println!("Generating random matrices...");
    let matrix_a = Box::new(Matrix::<SIZE, SIZE>::random());
    let matrix_b = Box::new(Matrix::<SIZE, SIZE>::random());

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
        let mut target = Box::new(Matrix::<SIZE, SIZE>::zeros());
        multiply(&matrix_a, &matrix_b, &mut target);
        start.elapsed().as_secs_f64() * 1000.0
    }).collect::<Vec<_>>();
    let forloop_avg = forloop_times.iter().sum::<f64>() / 3.0;

    // [B1] Multithread (50 threads)
    let b1_times = (0..3).map(|_| {
        let start = Instant::now();
        let mut target = Box::new(Matrix::<SIZE, SIZE>::zeros());
        multiply_parallel(&matrix_a, &matrix_b, 50, &mut target);
        start.elapsed().as_secs_f64() * 1000.0
    }).collect::<Vec<_>>();
    let b1_avg = b1_times.iter().sum::<f64>() / 3.0;

    // [B2] Multithread (10 threads)
    let b2_times = (0..3).map(|_| {
        let start = Instant::now();
        let mut target = Box::new(Matrix::<SIZE, SIZE>::zeros());
        multiply_parallel(&matrix_a, &matrix_b, 10, &mut target);
        start.elapsed().as_secs_f64() * 1000.0
    }).collect::<Vec<_>>();
    let b2_avg = b2_times.iter().sum::<f64>() / 3.0;

    println!(
        "| [A]         | 1              | {:21.6} | {:21.6} | {:21.6} | {:13.6} |",
        forloop_times[0], forloop_times[1], forloop_times[2], forloop_avg
    );
    println!(
        "| For-loops   | (50*50/thread) |                       |                       |                       |               |"
    );

    println!(
        "| [B1]        | 50             | {:21.6} | {:21.6} | {:21.6} | {:13.6} |",
        b1_times[0], b1_times[1], b1_times[2], b1_avg
    );
    println!(
        "| Multithread | (50*1/thread)  |                       |                       |                       |               |"
    );

    println!(
        "| [B2]        | 10             | {:21.6} | {:21.6} | {:21.6} | {:13.6} |",
        b2_times[0], b2_times[1], b2_times[2], b2_avg
    );
    println!(
        "| Multithread | (50*5/thread)  |                       |                       |                       |               |"
    );

    println!(
        "| Differences | 49             |                       |                       |                       | {:13.6} |",
        forloop_avg - b1_avg
    );
    println!(
        "| [B1 - A]    |                |                       |                       |                       |               |"
    );

    println!(
        "| Differences | 9              |                       |                       |                       | {:13.6} |",
        forloop_avg - b2_avg
    );
    println!(
        "| [B2 - A]    |                |                       |                       |                       |               |"
    );

    println!(
        "+-------------+----------------+-----------------------+-----------------------+-----------------------+---------------+"
    );
}
