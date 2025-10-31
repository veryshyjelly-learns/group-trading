// pub mod data;
pub mod backtesting;
pub mod clustering;
pub mod comparison;
pub mod performance;
pub mod plotting;
pub mod portfolio;
pub mod signals;
pub mod data;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

// Core data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    pub symbol: String,
    pub dates: Vec<NaiveDate>,
    pub prices: Vec<f64>,
    pub returns: Vec<f64>,
}

impl PriceData {
    pub fn new(symbol: String, dates: Vec<NaiveDate>, prices: Vec<f64>) -> Self {
        let returns = prices.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        Self {
            symbol,
            dates,
            prices,
            returns,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StockPair {
    pub stock: String,
    pub benchmark: String,
    pub hedge_ratio: f64,
    pub cluster_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub pair: String,
    pub stock_position: f64, // number of shares
    pub benchmark_position: f64,
    pub entry_date: NaiveDate,
    pub entry_spread: f64,
}

#[derive(Debug, Clone)]
pub enum Signal {
    Long,  // Buy stock, sell benchmark
    Short, // Sell stock, buy benchmark
    Exit,
    Hold,
}
