// use chrono::NaiveDate;
use group_trading::comparison::StrategyComparison;
use group_trading::{backtesting, comparison, plotting, signals, data};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading data...");
    let train_data = data::load_train_data("data/train")?;
    let test_data = data::load_test_data("data/test")?;
    let benchmark_symbol = "SPY".to_string();

    // Signal generator
    let signal_generator = Box::new(signals::ZScoreGenerator::new(60, 2.0, 0.5));

    // Backtest configuration
    let config = backtesting::BacktestConfig::default();

    // Run comparison
    println!("\nRunning strategy comparison...");
    let comparison = comparison::ComparisonEngine::compare_strategies(
        &train_data,
        &test_data,
        &benchmark_symbol,
        signal_generator,
        config,
    );

    // Print results
    comparison::ComparisonEngine::print_comparison(&comparison);

    // Generate individual performance reports
    let benchmark_data = test_data.get(&benchmark_symbol).unwrap();
    // let analyzer = performance::PerformanceAnalyzer::new(0.02);

    println!("\nGenerating output files...");

    // Vanilla strategy outputs
    plotting::PlotGenerator::generate_performance_report(
        &comparison.vanilla_result,
        benchmark_data,
        &comparison.vanilla_metrics,
        "output/vanilla_strategy_results.csv",
    )?;

    // Cluster strategy outputs
    plotting::PlotGenerator::generate_performance_report(
        &comparison.cluster_result,
        benchmark_data,
        &comparison.cluster_metrics,
        "output/cluster_strategy_results.csv",
    )?;

    // Combined comparison plot data
    generate_comparison_csv(&comparison, "output/strategy_comparison.csv")?;

    println!("Done! Check output/ directory for results.");

    Ok(())
}

fn generate_comparison_csv(
    comparison: &StrategyComparison,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(output_path)?;

    writeln!(file, "date,vanilla_equity,cluster_equity")?;

    // Align dates (assuming same dates in both)
    for ((date, vanilla_equity), (_, cluster_equity)) in comparison
        .vanilla_result
        .equity_curve
        .iter()
        .zip(comparison.cluster_result.equity_curve.iter())
    {
        writeln!(file, "{},{},{}", date, vanilla_equity, cluster_equity)?;
    }

    Ok(())
}


