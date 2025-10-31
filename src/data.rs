use std::collections::HashMap;
use std::error::Error;
use chrono::NaiveDate;
use serde::Deserialize;
use crate::PriceData;

#[derive(Debug, Deserialize)]
struct CsvRow {
    symbol: String,
    date: String,
    price: f64,
    #[serde(default)]
    ret: Option<f64>, // optional column if not precomputed
}


fn load_csv_data(path: &str) -> Result<HashMap<String, PriceData>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut data: HashMap<String, PriceData> = HashMap::new();

    for record in rdr.deserialize::<CsvRow>() {
        let row = record?;

        let date = NaiveDate::parse_from_str(&row.date, "%Y-%m-%d")?;

        let entry = data.entry(row.symbol.clone()).or_insert_with(|| PriceData {
            symbol: row.symbol.clone(),
            dates: Vec::new(),
            prices: Vec::new(),
            returns: Vec::new(),
        });

        entry.dates.push(date);
        entry.prices.push(row.price);
        if let Some(r) = row.ret {
            entry.returns.push(r);
        }
    }

    // if returns weren’t provided, compute them
    for entry in data.values_mut() {
        if entry.returns.is_empty() {
            entry.returns = entry
                .prices
                .windows(2)
                .map(|w| (w[1] / w[0]) - 1.0)
                .collect();
            entry.returns.insert(0, 0.0); // first return = 0.0
        }
    }

    Ok(data)
}

pub fn load_train_data(path: &str) -> Result<HashMap<String, PriceData>, Box<dyn Error>> {
    load_csv_data(path)
}

pub fn load_test_data(path: &str) -> Result<HashMap<String, PriceData>, Box<dyn Error>> {
    load_csv_data(path)
}
