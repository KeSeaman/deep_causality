use polars::prelude::*;
use anyhow::{Result, Context};
use tracing::info;

pub struct DataLoader;

impl DataLoader {
    /// Load a Parquet file into a Polars DataFrame
    pub fn load_parquet(path: &str) -> Result<DataFrame> {
        info!("Loading parquet file: {}", path);
        
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open file: {}", path))?;
        
        let df = ParquetReader::new(file)
            .finish()
            .with_context(|| format!("Failed to parse parquet: {}", path))?;
        
        info!("Loaded {} rows x {} columns", df.height(), df.width());
        Ok(df)
    }

    /// Load a CSV file into a Polars DataFrame
    pub fn load_csv(path: &str) -> Result<DataFrame> {
        info!("Loading CSV file: {}", path);
        
        let df = CsvReader::from_path(path)?
            .has_header(true)
            .finish()
            .with_context(|| format!("Failed to parse CSV: {}", path))?;
        
        info!("Loaded {} rows x {} columns", df.height(), df.width());
        Ok(df)
    }

    /// Filter DataFrame by a boolean column value
    pub fn filter_by_label(df: &DataFrame, column: &str, value: bool) -> Result<DataFrame> {
        let mask = df.column(column)?
            .bool()?
            .equal(value);
        
        df.filter(&mask)
            .context("Failed to filter DataFrame")
    }

    /// Get summary statistics for a DataFrame
    pub fn describe(df: &DataFrame) -> Result<DataFrame> {
        df.describe(None)
            .context("Failed to generate summary statistics")
    }

    /// Sample n rows from DataFrame (for testing with large datasets)
    pub fn sample(df: &DataFrame, n: usize, seed: Option<u64>) -> Result<DataFrame> {
        df.sample_n_literal(n, false, false, seed)
            .context("Failed to sample DataFrame")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader_exists() {
        // Basic existence test
        let _loader = DataLoader;
    }
}
