use crate::*;

pub trait SignalGenerator: Send + Sync {
    fn generate_signal(
        &mut self,
        pair: &StockPair,
        stock_data: &[f64],
        benchmark_data: &[f64],
        current_position: Option<&Position>,
    ) -> Signal;

    fn clone_box(&self) -> Box<dyn SignalGenerator>;
}

/// Z-Score based signal generator
pub struct ZScoreGenerator {
    pub lookback: usize,
    pub entry_threshold: f64,
    pub exit_threshold: f64,
}

impl ZScoreGenerator {
    pub fn new(lookback: usize, entry_threshold: f64, exit_threshold: f64) -> Self {
        Self {
            lookback,
            entry_threshold,
            exit_threshold,
        }
    }

    fn calculate_spread(
        &self,
        stock_prices: &[f64],
        benchmark_prices: &[f64],
        hedge_ratio: f64,
    ) -> Vec<f64> {
        stock_prices
            .iter()
            .zip(benchmark_prices.iter())
            .map(|(s, b)| s - hedge_ratio * b)
            .collect()
    }

    fn calculate_zscore(&self, spread: &[f64]) -> f64 {
        if spread.len() < self.lookback {
            return 0.0;
        }

        let window = &spread[spread.len() - self.lookback..];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        (spread[spread.len() - 1] - mean) / std_dev
    }
}

impl SignalGenerator for ZScoreGenerator {
    fn generate_signal(
        &mut self,
        pair: &StockPair,
        stock_data: &[f64],
        benchmark_data: &[f64],
        current_position: Option<&Position>,
    ) -> Signal {
        let spread = self.calculate_spread(stock_data, benchmark_data, pair.hedge_ratio);
        let z_score = self.calculate_zscore(&spread);

        match current_position {
            Some(_pos) => {
                // Exit logic
                if z_score.abs() < self.exit_threshold {
                    Signal::Exit
                } else {
                    Signal::Hold
                }
            }
            None => {
                // Entry logic
                if z_score > self.entry_threshold {
                    Signal::Short // Spread too high, short the spread
                } else if z_score < -self.entry_threshold {
                    Signal::Long // Spread too low, long the spread
                } else {
                    Signal::Hold
                }
            }
        }
    }

    fn clone_box(&self) -> Box<dyn SignalGenerator> {
        todo!()
    }
}

/// Bollinger Band based signal generator
pub struct BollingerBandGenerator {
    pub lookback: usize,
    pub num_std: f64,
}

impl SignalGenerator for BollingerBandGenerator {
    fn generate_signal(
        &mut self,
        pair: &StockPair,
        stock_data: &[f64],
        benchmark_data: &[f64],
        current_position: Option<&Position>,
    ) -> Signal {
        // Calculate spread
        let spread: Vec<f64> = stock_data
            .iter()
            .zip(benchmark_data.iter())
            .map(|(s, b)| s - pair.hedge_ratio * b)
            .collect();

        if spread.len() < self.lookback {
            return Signal::Hold;
        }

        let window = &spread[spread.len() - self.lookback..];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let std =
            (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64).sqrt();

        let upper_band = mean + self.num_std * std;
        let lower_band = mean - self.num_std * std;
        let current_spread = spread[spread.len() - 1];

        match current_position {
            Some(_) => {
                if current_spread >= mean - 0.5 * std && current_spread <= mean + 0.5 * std {
                    Signal::Exit
                } else {
                    Signal::Hold
                }
            }
            None => {
                if current_spread > upper_band {
                    Signal::Short
                } else if current_spread < lower_band {
                    Signal::Long
                } else {
                    Signal::Hold
                }
            }
        }
    }

    fn clone_box(&self) -> Box<dyn SignalGenerator> {
        todo!()
    }
}
