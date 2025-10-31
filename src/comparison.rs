use crate::backtesting::*;
use crate::performance::*;
use crate::portfolio::{ClusterBasedGenerator, PortfolioGenerator, VanillaFilterBasedGenerator};
use crate::signals::SignalGenerator;
use crate::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyComparison {
    pub vanilla_metrics: PerformanceMetrics,
    pub cluster_metrics: PerformanceMetrics,
    pub vanilla_result: BacktestResult,
    pub cluster_result: BacktestResult,
}

pub struct ComparisonEngine;

impl ComparisonEngine {
    pub fn compare_strategies(
        train_data: &HashMap<String, PriceData>,
        test_data: &HashMap<String, PriceData>,
        benchmark_symbol: &str,
        signal_generator: Box<dyn SignalGenerator>,
        config: BacktestConfig,
    ) -> StrategyComparison {
        println!("\n=== Running Vanilla Strategy ===");

        // Strategy 1: Vanilla (full dataset pairs)
        let vanilla_generator = VanillaFilterBasedGenerator::new(20);
        let vanilla_portfolio = vanilla_generator.generate(train_data, benchmark_symbol);
        println!("Vanilla portfolio: {} pairs", vanilla_portfolio.len());

        let mut vanilla_backtest = BacktestEngine::new(
            config.clone(),
            vanilla_portfolio,
            signal_generator.clone_box(),
        );
        let vanilla_result = vanilla_backtest.run(test_data);

        println!("\n=== Running Cluster-Based Strategy ===");

        // Strategy 2: Cluster-based
        let cluster_generator = ClusterBasedGenerator::new(
            252,  // 1 year lookback
            3,    // 3 pairs per cluster
            None, // Use eRank to determine clusters
        );
        let cluster_portfolio = cluster_generator.generate(train_data, benchmark_symbol);
        println!("Cluster portfolio: {} pairs", cluster_portfolio.len());

        let mut cluster_backtest = BacktestEngine::new(
            config.clone(),
            cluster_portfolio,
            signal_generator.clone_box(),
        );
        let cluster_result = cluster_backtest.run(test_data);

        // Calculate performance metrics
        let benchmark_data = test_data.get(benchmark_symbol).unwrap();
        let analyzer = PerformanceAnalyzer::new(0.02);

        let vanilla_metrics = analyzer.analyze(&vanilla_result, benchmark_data);
        let cluster_metrics = analyzer.analyze(&cluster_result, benchmark_data);

        StrategyComparison {
            vanilla_metrics,
            cluster_metrics,
            vanilla_result,
            cluster_result,
        }
    }

    pub fn print_comparison(comparison: &StrategyComparison) {
        println!("\n{}", "=".repeat(70));
        println!("STRATEGY COMPARISON RESULTS");
        println!("{}\n", "=".repeat(70));

        println!(
            "{:<30} {:>15} {:>15} {:>15}",
            "Metric", "Vanilla", "Cluster-Based", "Difference"
        );
        println!("{}", "-".repeat(70));

        Self::print_metric_row(
            "Total Return (%)",
            comparison.vanilla_metrics.total_return * 100.0,
            comparison.cluster_metrics.total_return * 100.0,
        );

        Self::print_metric_row(
            "Annualized Return (%)",
            comparison.vanilla_metrics.annualized_return * 100.0,
            comparison.cluster_metrics.annualized_return * 100.0,
        );

        Self::print_metric_row(
            "Sharpe Ratio",
            comparison.vanilla_metrics.sharpe_ratio,
            comparison.cluster_metrics.sharpe_ratio,
        );

        Self::print_metric_row(
            "Sortino Ratio",
            comparison.vanilla_metrics.sortino_ratio,
            comparison.cluster_metrics.sortino_ratio,
        );

        Self::print_metric_row(
            "Max Drawdown (%)",
            comparison.vanilla_metrics.max_drawdown * 100.0,
            comparison.cluster_metrics.max_drawdown * 100.0,
        );

        Self::print_metric_row(
            "Win Rate (%)",
            comparison.vanilla_metrics.win_rate * 100.0,
            comparison.cluster_metrics.win_rate * 100.0,
        );

        Self::print_metric_row(
            "Number of Trades",
            comparison.vanilla_metrics.num_trades as f64,
            comparison.cluster_metrics.num_trades as f64,
        );

        Self::print_metric_row(
            "Alpha (%)",
            comparison.vanilla_metrics.alpha * 100.0,
            comparison.cluster_metrics.alpha * 100.0,
        );

        Self::print_metric_row(
            "Beta",
            comparison.vanilla_metrics.beta,
            comparison.cluster_metrics.beta,
        );

        Self::print_metric_row(
            "Information Ratio",
            comparison.vanilla_metrics.information_ratio,
            comparison.cluster_metrics.information_ratio,
        );

        Self::print_metric_row(
            "Calmar Ratio",
            comparison.vanilla_metrics.calmar_ratio,
            comparison.cluster_metrics.calmar_ratio,
        );

        println!("{}", "=".repeat(70));
    }

    fn print_metric_row(name: &str, vanilla: f64, cluster: f64) {
        let diff = cluster - vanilla;
        let diff_pct = if vanilla.abs() > 1e-10 {
            (diff / vanilla.abs()) * 100.0
        } else {
            0.0
        };

        println!(
            "{:<30} {:>15.3} {:>15.3} {:>15.3} ({:+.1}%)",
            name, vanilla, cluster, diff, diff_pct
        );
    }
}
