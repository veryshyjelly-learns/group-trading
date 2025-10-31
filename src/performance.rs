use crate::backtesting::BacktestResult;
use crate::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: usize,
    pub alpha: f64,
    pub beta: f64,
    pub information_ratio: f64,
    pub calmar_ratio: f64,
}

pub struct PerformanceAnalyzer {
    risk_free_rate: f64,
}

impl PerformanceAnalyzer {
    pub fn new(risk_free_rate: f64) -> Self {
        Self { risk_free_rate }
    }

    pub fn analyze(
        &self,
        strategy_result: &BacktestResult,
        benchmark_data: &PriceData,
    ) -> PerformanceMetrics {
        let returns = self.calculate_returns(&strategy_result.equity_curve);
        let benchmark_returns = &benchmark_data.returns;

        // Align benchmark returns with strategy dates
        let aligned_benchmark = self.align_benchmark_returns(
            &strategy_result.equity_curve,
            &benchmark_data.dates,
            benchmark_returns,
        );

        let total_return = (strategy_result.final_capital / strategy_result.initial_capital) - 1.0;
        let num_years = strategy_result.equity_curve.len() as f64 / 252.0;
        let annualized_return = ((1.0 + total_return).powf(1.0 / num_years)) - 1.0;

        let sharpe_ratio = self.calculate_sharpe_ratio(&returns);
        let sortino_ratio = self.calculate_sortino_ratio(&returns);
        let max_drawdown = self.calculate_max_drawdown(&strategy_result.equity_curve);

        let win_rate = strategy_result
            .trades
            .iter()
            .filter(|t| t.pnl > 0.0)
            .count() as f64
            / strategy_result.trades.len().max(1) as f64;

        let (alpha, beta) = self.calculate_alpha_beta(&returns, &aligned_benchmark);
        let information_ratio = self.calculate_information_ratio(&returns, &aligned_benchmark);
        let calmar_ratio = annualized_return / max_drawdown.abs();

        PerformanceMetrics {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            num_trades: strategy_result.trades.len(),
            alpha,
            beta,
            information_ratio,
            calmar_ratio,
        }
    }

    fn calculate_returns(&self, equity_curve: &[(NaiveDate, f64)]) -> Vec<f64> {
        equity_curve
            .windows(2)
            .map(|w| (w[1].1 / w[0].1) - 1.0)
            .collect()
    }

    fn align_benchmark_returns(
        &self,
        equity_dates: &[(NaiveDate, f64)],
        benchmark_dates: &[NaiveDate],
        benchmark_returns: &[f64],
    ) -> Vec<f64> {
        let benchmark_map: HashMap<NaiveDate, f64> = benchmark_dates
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(d, r)| (*d, *r))
            .collect();

        equity_dates
            .windows(2)
            .map(|w| *benchmark_map.get(&w[1].0).unwrap_or(&0.0))
            .collect()
    }

    fn calculate_sharpe_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        let daily_rf = self.risk_free_rate / 252.0;
        let excess_return = mean_return - daily_rf;

        (excess_return / std_dev) * (252.0_f64).sqrt()
    }

    fn calculate_sortino_ratio(&self, returns: &[f64]) -> f64 {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let daily_rf = self.risk_free_rate / 252.0;

        let downside_returns: Vec<f64> = returns
            .iter()
            .map(|r| if *r < daily_rf { r - daily_rf } else { 0.0 })
            .collect();

        let downside_variance =
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std == 0.0 {
            return 0.0;
        }

        ((mean_return - daily_rf) / downside_std) * (252.0_f64).sqrt()
    }

    fn calculate_max_drawdown(&self, equity_curve: &[(NaiveDate, f64)]) -> f64 {
        let mut max_drawdown = 0.0;
        let mut peak = equity_curve[0].1;

        for (_, value) in equity_curve.iter() {
            if *value > peak {
                peak = *value;
            }
            let drawdown = (*value - peak) / peak;
            if drawdown < max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    fn calculate_alpha_beta(
        &self,
        strategy_returns: &[f64],
        benchmark_returns: &[f64],
    ) -> (f64, f64) {
        let n = strategy_returns.len().min(benchmark_returns.len()) as f64;
        let mean_strategy = strategy_returns.iter().sum::<f64>() / n;
        let mean_benchmark = benchmark_returns.iter().sum::<f64>() / n;

        let cov: f64 = strategy_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(s, b)| (s - mean_strategy) * (b - mean_benchmark))
            .sum::<f64>()
            / n;

        let var_benchmark: f64 = benchmark_returns
            .iter()
            .map(|b| (b - mean_benchmark).powi(2))
            .sum::<f64>()
            / n;

        let beta = cov / var_benchmark;
        let alpha = (mean_strategy - self.risk_free_rate / 252.0)
            - beta * (mean_benchmark - self.risk_free_rate / 252.0);

        // Annualize alpha
        (alpha * 252.0, beta)
    }

    fn calculate_information_ratio(
        &self,
        strategy_returns: &[f64],
        benchmark_returns: &[f64],
    ) -> f64 {
        let tracking_error: Vec<f64> = strategy_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(s, b)| s - b)
            .collect();

        let mean_te = tracking_error.iter().sum::<f64>() / tracking_error.len() as f64;
        let std_te = (tracking_error
            .iter()
            .map(|te| (te - mean_te).powi(2))
            .sum::<f64>()
            / tracking_error.len() as f64)
            .sqrt();

        if std_te == 0.0 {
            return 0.0;
        }

        (mean_te / std_te) * (252.0_f64).sqrt()
    }
}
