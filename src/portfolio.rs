use crate::clustering::*;
use crate::*;
use std::collections::HashMap;

pub trait PortfolioGenerator: Send + Sync {
    fn generate(
        &self,
        train_data: &HashMap<String, PriceData>,
        benchmark_symbol: &str,
    ) -> Vec<StockPair>;
}

/// Vanilla strategy: find pairs across full dataset
pub struct VanillaFilterBasedGenerator {
    pub filter: PairFilter,
    pub max_pairs: usize,
}

impl VanillaFilterBasedGenerator {
    pub fn new(max_pairs: usize) -> Self {
        Self {
            filter: PairFilter::new(),
            max_pairs,
        }
    }
}

impl PortfolioGenerator for VanillaFilterBasedGenerator {
    fn generate(
        &self,
        train_data: &HashMap<String, PriceData>,
        benchmark_symbol: &str,
    ) -> Vec<StockPair> {
        let benchmark = train_data
            .get(benchmark_symbol)
            .expect("Benchmark not found");

        let mut candidates = Vec::new();

        // Test all stocks against benchmark
        for (symbol, stock_data) in train_data.iter() {
            if symbol == benchmark_symbol {
                continue;
            }

            let min_len = stock_data.returns.len().min(benchmark.returns.len());
            let stock_rets = &stock_data.returns[..min_len];
            let bench_rets = &benchmark.returns[..min_len];

            let result = self.filter.evaluate(stock_rets, bench_rets);

            if result.passes {
                let hedge_ratio = calculate_hedge_ratio(stock_rets, bench_rets);

                candidates.push((
                    symbol.clone(),
                    result.hurst_exponent,
                    result.half_life_days,
                    hedge_ratio,
                ));
            }
        }

        // Sort and take top pairs
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap()
                .then(a.2.partial_cmp(&b.2).unwrap())
        });

        candidates
            .into_iter()
            .take(self.max_pairs)
            .map(|(symbol, _, _, hedge_ratio)| StockPair {
                stock: symbol,
                benchmark: benchmark_symbol.to_string(),
                hedge_ratio,
                cluster_id: None,
            })
            .collect()
    }
}

/// Cluster-based strategy: cluster first, then find pairs within clusters
pub struct ClusterBasedGenerator {
    pub filter: PairFilter,
    pub clustering: StatisticalClustering,
    pub pairs_per_cluster: usize,
    pub num_clusters: Option<usize>, // If None, use eRank
}

impl ClusterBasedGenerator {
    pub fn new(lookback: usize, pairs_per_cluster: usize, num_clusters: Option<usize>) -> Self {
        Self {
            filter: PairFilter::new(),
            clustering: StatisticalClustering::new(lookback, 100), // 100 samplings
            pairs_per_cluster,
            num_clusters,
        }
    }
}

impl PortfolioGenerator for ClusterBasedGenerator {
    fn generate(
        &self,
        train_data: &HashMap<String, PriceData>,
        _benchmark_symbol: &str,
    ) -> Vec<StockPair> {
        // Step 1: Determine number of clusters
        let k = match self.num_clusters {
            Some(k) => k,
            None => self.clustering.determine_num_clusters(train_data),
        };

        println!("Clustering into {} clusters", k);

        // Step 2: Cluster stocks
        let cluster_assignment = self.clustering.cluster_stocks(train_data, k);

        println!(
            "Cluster sizes: {:?}",
            cluster_assignment
                .cluster_to_stocks
                .values()
                .map(|v| v.len())
                .collect::<Vec<_>>()
        );

        // Step 3: Find pairs within each cluster
        let mut all_pairs = Vec::new();

        for (cluster_id, stocks) in cluster_assignment.cluster_to_stocks.iter() {
            if stocks.len() < 2 {
                continue; // Need at least 2 stocks to form pairs
            }

            let mut cluster_candidates = Vec::new();

            // Find best pairs within this cluster
            for i in 0..stocks.len() {
                for j in (i + 1)..stocks.len() {
                    let stock_a = &stocks[i];
                    let stock_b = &stocks[j];

                    let data_a = match train_data.get(stock_a) {
                        Some(d) => d,
                        None => continue,
                    };
                    let data_b = match train_data.get(stock_b) {
                        Some(d) => d,
                        None => continue,
                    };

                    let min_len = data_a.returns.len().min(data_b.returns.len());
                    let rets_a = &data_a.returns[..min_len];
                    let rets_b = &data_b.returns[..min_len];

                    let result = self.filter.evaluate(rets_a, rets_b);

                    if result.passes {
                        let hedge_ratio = calculate_hedge_ratio(rets_a, rets_b);

                        cluster_candidates.push((
                            stock_a.clone(),
                            stock_b.clone(),
                            result.hurst_exponent,
                            result.half_life_days,
                            hedge_ratio,
                        ));
                    }
                }
            }

            // Sort and take top pairs from this cluster
            cluster_candidates.sort_by(|a, b| {
                a.2.partial_cmp(&b.2)
                    .unwrap()
                    .then(a.3.partial_cmp(&b.3).unwrap())
            });

            for (stock, bench, _, _, hedge_ratio) in
                cluster_candidates.into_iter().take(self.pairs_per_cluster)
            {
                all_pairs.push(StockPair {
                    stock,
                    benchmark: bench,
                    hedge_ratio,
                    cluster_id: Some(*cluster_id),
                });
            }
        }

        println!("Found {} pairs across all clusters", all_pairs.len());

        all_pairs
    }
}

fn calculate_hedge_ratio(stock_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let n = stock_returns.len() as f64;
    let mean_x = benchmark_returns.iter().sum::<f64>() / n;
    let mean_y = stock_returns.iter().sum::<f64>() / n;

    let cov: f64 = benchmark_returns
        .iter()
        .zip(stock_returns.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();

    let var_x: f64 = benchmark_returns
        .iter()
        .map(|x| (x - mean_x).powi(2))
        .sum::<f64>();

    cov / var_x
}

// [PairFilter implementation from previous code]
pub struct PairFilter {
    formation_days: usize,
    min_half_life: f64,
    max_half_life: f64,
    max_hurst: f64,
    min_rmr2: f64,
}

impl PairFilter {
    pub fn new() -> Self {
        Self {
            formation_days: 252, // 1 year
            min_half_life: 1.0,
            max_half_life: 60.0,
            max_hurst: 0.5,
            min_rmr2: 0.5,
        }
    }

    /// Main filter: returns whether stock-benchmark pair is acceptable
    pub fn evaluate(&self, stock_returns: &[f64], benchmark_returns: &[f64]) -> FilterResult {
        assert_eq!(stock_returns.len(), benchmark_returns.len());
        assert!(stock_returns.len() >= self.formation_days);

        // 1. SSD Test
        let ssd_result = self.test_ssd(stock_returns, benchmark_returns);

        // 2. Compute spread (stock - beta*benchmark)
        let (spread, _beta) = self.compute_spread(stock_returns, benchmark_returns);

        // 3. Fractional Cointegration
        let frac_result = self.test_fractional_cointegration(&spread);

        // 4. Half-life
        let half_life = self.compute_half_life(&spread);

        // 5. Hurst Exponent
        let hurst = self.compute_hurst_dfa(&spread);

        // Decision logic
        let passes = ssd_result
            && frac_result.passes
            && half_life >= self.min_half_life
            && half_life <= self.max_half_life
            && hurst < self.max_hurst;

        let reason = if !passes {
            if !ssd_result {
                "Failed SSD: stock does not dominate benchmark".to_string()
            } else if !frac_result.passes {
                format!("Failed fractional coint: R^2_MR = {:.3}", frac_result.rmr2)
            } else if half_life < self.min_half_life {
                format!("Half-life too short: {:.2} days", half_life)
            } else if half_life > self.max_half_life {
                format!("Half-life too long: {:.2} days", half_life)
            } else {
                format!("Hurst too high: {:.3} (not mean-reverting)", hurst)
            }
        } else {
            "All filters passed".to_string()
        };

        FilterResult {
            passes,
            ssd_dominates: ssd_result,
            frac_coint_rmr2: frac_result.rmr2,
            half_life_days: half_life,
            hurst_exponent: hurst,
            reason,
        }
    }

    /// Test Second-Order Stochastic Dominance (SSD)
    /// For Gaussian: X dominates Y if mu_X >= mu_Y and sigma_X <= sigma_Y
    fn test_ssd(&self, stock_returns: &[f64], benchmark_returns: &[f64]) -> bool {
        let stock_mean = stock_returns.iter().sum::<f64>() / stock_returns.len() as f64;
        let bench_mean = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;

        let stock_var = stock_returns.iter()
            .map(|x| (x - stock_mean).powi(2))
            .sum::<f64>() / (stock_returns.len() - 1) as f64;
        let bench_var = benchmark_returns.iter()
            .map(|x| (x - bench_mean).powi(2))
            .sum::<f64>() / (benchmark_returns.len() - 1) as f64;

        let stock_std = stock_var.sqrt();
        let bench_std = bench_var.sqrt();

        // SSD for Gaussian: higher mean, lower volatility
        stock_mean >= bench_mean && stock_std <= bench_std
    }

    /// Compute spread using OLS regression
    fn compute_spread(&self, stock: &[f64], benchmark: &[f64]) -> (Vec<f64>, f64) {
        let n = stock.len() as f64;
        let mean_x = benchmark.iter().sum::<f64>() / n;
        let mean_y = stock.iter().sum::<f64>() / n;

        let cov_xy: f64 = benchmark.iter().zip(stock.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>() / (n - 1.0);

        let var_x: f64 = benchmark.iter()
            .map(|x| (x - mean_x).powi(2))
            .sum::<f64>() / (n - 1.0);

        let beta = cov_xy / var_x;
        let alpha = mean_y - beta * mean_x;

        let spread: Vec<f64> = stock.iter().zip(benchmark.iter())
            .map(|(y, x)| y - (alpha + beta * x))
            .collect();

        (spread, beta)
    }

    /// Test fractional cointegration
    /// Implements the partial cointegration test from Clegg & Krauss (2018)
    fn test_fractional_cointegration(&self, spread: &[f64]) -> FracCointResult {
        // Fit AR(1) to spread
        let (rho, sigma_m) = self.fit_ar1(spread);

        // Estimate random walk component variance
        let innovations: Vec<f64> = spread.windows(2)
            .map(|w| w[1] - rho * w[0])
            .collect();

        let sigma_r_sq = innovations.iter()
            .map(|e| e.powi(2))
            .sum::<f64>() / innovations.len() as f64;

        let sigma_m_sq = sigma_m.powi(2);

        // Calculate R^2_MR (proportion of mean-reverting variance)
        let rmr2 = (2.0 * sigma_m_sq) / (2.0 * sigma_m_sq + (1.0 + rho) * sigma_r_sq);

        FracCointResult {
            passes: rmr2 > self.min_rmr2 && rho > 0.5,
            rmr2,
            rho,
        }
    }

    /// Fit AR(1) model: y_t = rho * y_{t-1} + epsilon_t
    fn fit_ar1(&self, series: &[f64]) -> (f64, f64) {
        let y_lag: Vec<f64> = series[..series.len()-1].to_vec();
        let y_curr: Vec<f64> = series[1..].to_vec();

        let n = y_lag.len() as f64;
        let mean_lag = y_lag.iter().sum::<f64>() / n;
        let mean_curr = y_curr.iter().sum::<f64>() / n;

        let cov: f64 = y_lag.iter().zip(y_curr.iter())
            .map(|(x, y)| (x - mean_lag) * (y - mean_curr))
            .sum::<f64>();

        let var_lag: f64 = y_lag.iter()
            .map(|x| (x - mean_lag).powi(2))
            .sum::<f64>();

        let rho = cov / var_lag;

        // Residual variance
        let residuals: Vec<f64> = y_lag.iter().zip(y_curr.iter())
            .map(|(x, y)| y - rho * x)
            .collect();

        let sigma = (residuals.iter().map(|r| r.powi(2)).sum::<f64>() / n).sqrt();

        (rho, sigma)
    }

    /// Compute half-life of mean reversion
    /// From AR(1): delta_y = lambda * y_{t-1} + epsilon
    fn compute_half_life(&self, spread: &[f64]) -> f64 {
        let delta_y: Vec<f64> = spread.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let y_lag: Vec<f64> = spread[..spread.len()-1].to_vec();

        let n = y_lag.len() as f64;
        let mean_y = y_lag.iter().sum::<f64>() / n;
        let mean_delta = delta_y.iter().sum::<f64>() / n;

        let cov: f64 = y_lag.iter().zip(delta_y.iter())
            .map(|(y, d)| (y - mean_y) * (d - mean_delta))
            .sum::<f64>();

        let var_y: f64 = y_lag.iter()
            .map(|y| (y - mean_y).powi(2))
            .sum::<f64>();

        let lambda = cov / var_y;

        // Half-life = -ln(2) / ln(1 + lambda)
        if lambda >= 0.0 {
            return f64::INFINITY; // No mean reversion
        }

        -f64::ln(2.0) / f64::ln(1.0 + lambda)
    }

    /// Compute Hurst exponent using Detrended Fluctuation Analysis (DFA)
    fn compute_hurst_dfa(&self, series: &[f64]) -> f64 {
        let mean = series.iter().sum::<f64>() / series.len() as f64;

        // 1. Integrate the series
        let mut integrated = Vec::with_capacity(series.len());
        let mut cum_sum = 0.0;
        for &val in series {
            cum_sum += val - mean;
            integrated.push(cum_sum);
        }

        // 2. Test multiple box sizes
        let min_box = 10;
        let max_box = series.len() / 4;
        let num_boxes = 20;

        let box_sizes: Vec<usize> = (0..num_boxes)
            .map(|i| {
                let log_min = (min_box as f64).ln();
                let log_max = (max_box as f64).ln();
                (log_min + (log_max - log_min) * i as f64 / (num_boxes - 1) as f64).exp() as usize
            })
            .collect();

        let mut log_n = Vec::new();
        let mut log_f = Vec::new();

        for &box_size in &box_sizes {
            let num_segments = integrated.len() / box_size;
            if num_segments < 2 {
                continue;
            }

            let mut fluctuations = Vec::new();

            for seg in 0..num_segments {
                let start = seg * box_size;
                let end = start + box_size;
                let segment = &integrated[start..end];

                // Fit linear trend
                let (slope, intercept) = self.linear_fit(segment);

                // Calculate detrended fluctuation
                let fluct: f64 = segment.iter().enumerate()
                    .map(|(i, &y)| {
                        let trend = slope * i as f64 + intercept;
                        (y - trend).powi(2)
                    })
                    .sum::<f64>() / box_size as f64;

                fluctuations.push(fluct);
            }

            let f_n = fluctuations.iter().sum::<f64>() / fluctuations.len() as f64;
            let f_n = f_n.sqrt();

            log_n.push((box_size as f64).ln());
            log_f.push(f_n.ln());
        }

        // 3. Hurst = slope of log(F(n)) vs log(n)
        let n_mean = log_n.iter().sum::<f64>() / log_n.len() as f64;
        let f_mean = log_f.iter().sum::<f64>() / log_f.len() as f64;

        let cov: f64 = log_n.iter().zip(log_f.iter())
            .map(|(n, f)| (n - n_mean) * (f - f_mean))
            .sum();

        let var_n: f64 = log_n.iter()
            .map(|n| (n - n_mean).powi(2))
            .sum();

        cov / var_n
    }

    /// Linear regression helper
    fn linear_fit(&self, data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let x_vals: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();

        let mean_x = x_vals.iter().sum::<f64>() / n;
        let mean_y = data.iter().sum::<f64>() / n;

        let cov: f64 = x_vals.iter().zip(data.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();

        let var_x: f64 = x_vals.iter()
            .map(|x| (x - mean_x).powi(2))
            .sum();

        let slope = cov / var_x;
        let intercept = mean_y - slope * mean_x;

        (slope, intercept)
    }
}

#[derive(Debug)]
struct FracCointResult {
    passes: bool,
    rmr2: f64,
    #[allow(dead_code)]
    rho: f64,
}

#[derive(Debug)]
pub struct FilterResult {
    pub passes: bool,
    pub ssd_dominates: bool,
    pub frac_coint_rmr2: f64,
    pub half_life_days: f64,
    pub hurst_exponent: f64,
    pub reason: String,
}
