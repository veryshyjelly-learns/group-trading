use crate::signals::SignalGenerator;
use crate::*;
use std::collections::HashMap;

#[derive(Copy, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub position_size: f64,    // Fraction of capital per trade
    pub transaction_cost: f64, // Basis points
    pub slippage: f64,         // Basis points
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            transaction_cost: 10.0, // 10 bps
            slippage: 5.0,          // 5 bps
        }
    }
}

pub struct BacktestEngine {
    config: BacktestConfig,
    portfolio: Vec<StockPair>,
    signal_generator: Box<dyn SignalGenerator>,
}

impl BacktestEngine {
    pub fn new(
        config: BacktestConfig,
        portfolio: Vec<StockPair>,
        signal_generator: Box<dyn SignalGenerator>,
    ) -> Self {
        Self {
            config,
            portfolio,
            signal_generator,
        }
    }

    pub fn run(&mut self, test_data: &HashMap<String, PriceData>) -> BacktestResult {
        let mut equity_curve = Vec::new();
        let mut positions: HashMap<String, Position> = HashMap::new();
        let mut trades = Vec::new();
        let mut current_capital = self.config.initial_capital;

        // Get all dates from first stock
        let dates = &test_data.values().next().unwrap().dates;

        for (day_idx, date) in dates.iter().enumerate() {
            let mut daily_pnl = 0.0;

            // Process each pair
            for pair in &self.portfolio {
                let stock_data = test_data.get(&pair.stock).unwrap();
                let benchmark_data = test_data.get(&pair.benchmark).unwrap();

                // Get price history up to current day
                let stock_prices = &stock_data.prices[..=day_idx];
                let benchmark_prices = &benchmark_data.prices[..=day_idx];

                let pair_key = format!("{}_{}", pair.stock, pair.benchmark);
                let current_position = positions.get(&pair_key);

                // Generate signal
                let signal = self.signal_generator.generate_signal(
                    pair,
                    stock_prices,
                    benchmark_prices,
                    current_position,
                );

                // Execute signal
                match signal {
                    Signal::Long => {
                        if current_position.is_none() {
                            // Enter long position
                            let position_value = current_capital * self.config.position_size;
                            let stock_shares = position_value / stock_prices[day_idx];
                            let benchmark_shares =
                                -(position_value * pair.hedge_ratio) / benchmark_prices[day_idx];

                            let cost = self.calculate_transaction_cost(
                                position_value * (1.0 + pair.hedge_ratio),
                            );
                            current_capital -= cost;

                            positions.insert(
                                pair_key.clone(),
                                Position {
                                    pair: pair_key.clone(),
                                    stock_position: stock_shares,
                                    benchmark_position: benchmark_shares,
                                    entry_date: *date,
                                    entry_spread: stock_prices[day_idx]
                                        - pair.hedge_ratio * benchmark_prices[day_idx],
                                },
                            );
                        }
                    }
                    Signal::Short => {
                        if current_position.is_none() {
                            // Enter short position
                            let position_value = current_capital * self.config.position_size;
                            let stock_shares = -position_value / stock_prices[day_idx];
                            let benchmark_shares =
                                (position_value * pair.hedge_ratio) / benchmark_prices[day_idx];

                            let cost = self.calculate_transaction_cost(
                                position_value * (1.0 + pair.hedge_ratio),
                            );
                            current_capital -= cost;

                            positions.insert(
                                pair_key.clone(),
                                Position {
                                    pair: pair_key.clone(),
                                    stock_position: stock_shares,
                                    benchmark_position: benchmark_shares,
                                    entry_date: *date,
                                    entry_spread: stock_prices[day_idx]
                                        - pair.hedge_ratio * benchmark_prices[day_idx],
                                },
                            );
                        }
                    }
                    Signal::Exit => {
                        if let Some(pos) = positions.remove(&pair_key) {
                            // Calculate PnL
                            let stock_pnl = pos.stock_position
                                * (stock_prices[day_idx] - stock_prices[day_idx - 1]);
                            let benchmark_pnl = pos.benchmark_position
                                * (benchmark_prices[day_idx] - benchmark_prices[day_idx - 1]);
                            let trade_pnl = stock_pnl + benchmark_pnl;

                            let exit_value = pos.stock_position.abs() * stock_prices[day_idx]
                                + pos.benchmark_position.abs() * benchmark_prices[day_idx];
                            let cost = self.calculate_transaction_cost(exit_value);

                            daily_pnl += trade_pnl - cost;
                            current_capital += trade_pnl - cost;

                            trades.push(Trade {
                                pair: pair_key.clone(),
                                entry_date: pos.entry_date,
                                exit_date: *date,
                                pnl: trade_pnl - cost,
                            });
                        }
                    }
                    Signal::Hold => {
                        // Calculate mark-to-market PnL for open positions
                        if let Some(pos) = positions.get(&pair_key) {
                            if day_idx > 0 {
                                let stock_pnl = pos.stock_position
                                    * (stock_prices[day_idx] - stock_prices[day_idx - 1]);
                                let benchmark_pnl = pos.benchmark_position
                                    * (benchmark_prices[day_idx] - benchmark_prices[day_idx - 1]);
                                daily_pnl += stock_pnl + benchmark_pnl;
                            }
                        }
                    }
                }
            }

            // Update equity curve
            current_capital += daily_pnl;
            equity_curve.push((*date, current_capital));
        }

        BacktestResult {
            equity_curve,
            trades,
            final_capital: current_capital,
            initial_capital: self.config.initial_capital,
        }
    }

    fn calculate_transaction_cost(&self, value: f64) -> f64 {
        value * (self.config.transaction_cost + self.config.slippage) / 10_000.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub pair: String,
    pub entry_date: NaiveDate,
    pub exit_date: NaiveDate,
    pub pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub equity_curve: Vec<(NaiveDate, f64)>,
    pub trades: Vec<Trade>,
    pub final_capital: f64,
    pub initial_capital: f64,
}
