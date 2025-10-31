use crate::backtesting::BacktestResult;
use crate::performance::PerformanceMetrics;
use crate::*;
use std::fs::File;
use std::io::Write;

pub struct PlotGenerator;

impl PlotGenerator {
    pub fn generate_performance_report(
        strategy_result: &BacktestResult,
        benchmark_data: &PriceData,
        metrics: &PerformanceMetrics,
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Generate CSV for plotting with Python/external tools
        let mut file = File::create(output_path)?;

        writeln!(file, "date,strategy_equity,benchmark_value")?;

        let initial_benchmark_price = benchmark_data.prices[0];
        let benchmark_map: std::collections::HashMap<NaiveDate, f64> = benchmark_data
            .dates
            .iter()
            .zip(benchmark_data.prices.iter())
            .map(|(d, p)| (*d, *p))
            .collect();

        for (date, equity) in &strategy_result.equity_curve {
            let benchmark_price = benchmark_map.get(date).unwrap_or(&initial_benchmark_price);
            let benchmark_equity =
                strategy_result.initial_capital * (benchmark_price / initial_benchmark_price);

            writeln!(file, "{},{},{}", date, equity, benchmark_equity)?;
        }

        // Write metrics summary
        let metrics_path = output_path.replace(".csv", "_metrics.json");
        let metrics_json = serde_json::to_string_pretty(metrics)?;
        std::fs::write(metrics_path, metrics_json)?;

        Ok(())
    }

    pub fn generate_drawdown_csv(
        equity_curve: &[(NaiveDate, f64)],
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(output_path)?;
        writeln!(file, "date,equity,drawdown")?;

        let mut peak = equity_curve[0].1;
        for (date, value) in equity_curve {
            if *value > peak {
                peak = *value;
            }
            let drawdown = (*value - peak) / peak;
            writeln!(file, "{},{},{}", date, value, drawdown)?;
        }

        Ok(())
    }
}
