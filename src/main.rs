// EXternal crates
use argparse::{ArgumentParser, StoreTrue};
// Local imports
mod benchmarks;
use benchmarks::{sensitibility_benchmarks, single_test_benchmark};

// Main function. Works for three different things:
//
// 1. To ensure that the code is being compiled correctly
// 2. To run the benchmarks. Those benchmarks can be:
//     - Sensitivity analysis of the parameters
//     - Single SDAO test on the benchmark functions
//     - Comparing SDAO against other functions.
fn main() {
    println!("======================");
    println!("SDAO Benchmarking...");
    println!("======================");

    // Define the variable to know the type of benchmark to run
    let mut sensitivity = false;
    let mut single_test = false;
    let mut compare = false;
    // Instanciate the argument parser
    let mut ap = ArgumentParser::new();
    ap.set_description("Select benchmark type to test the SDAO algorithm");
    // Add the arguments to the parser
    ap.refer(&mut sensitivity).add_option(
        &["-s", "--sensitivity"],
        StoreTrue,
        "Run the sensitivity analysis",
    );

    ap.refer(&mut single_test).add_option(
        &["-t", "--test"],
        StoreTrue,
        "Run a single test on the benchmark functions",
    );

    ap.refer(&mut compare).add_option(
        &["-c", "--compare"],
        StoreTrue,
        "Compare the SDAO against other optimization algorithms",
    );

    // Parse the arguments
    ap.parse_args_or_exit();
    drop(ap); //* Ensure the mutable borrow ends here

    // Check the type of benchmark to run
    if sensitivity {
        println!("Running sensitivity analysis...");
        // Run the sensitivity analysis
        sensitibility_benchmarks();
    } else if single_test {
        println!("Running single test...");
        // Run the single test
        single_test_benchmark();
    } else if compare {
        println!("Running comparison against algorithms...");
        // Run the comparison
        // compare_benchmarks();
    } else {
        println!("No benchmark selected. Please select one of the following options:");
        println!("\t-s, --sensitivity\tRun the sensitivity analysis");
        println!("\t-t, --test\tRun a single test on the benchmark functions");
        println!("\t-c, --compare\tCompare the SDAO against other optimization algorithms");
    }
}
