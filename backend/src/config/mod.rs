use serde::Deserialize;
use std::fs;
use anyhow::{Context, Result};

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub data: DataConfig,
    pub experiment: ExperimentConfig,
    pub causality: CausalityConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DataConfig {
    pub train_path: String,
    pub test_path: String,
    pub validation_path: String,
    pub sepsis_subset_path: String,
    pub non_sepsis_subset_path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExperimentConfig {
    pub target_column: String,
    pub patient_id_column: String,
    pub time_column: String,
    pub test_size: f64,
    pub random_seed: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CausalityConfig {
    pub significance_threshold: f64,
    pub max_features: usize,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file at {}", path))?;
        let config: Config = toml::from_str(&content)
            .context("Failed to parse config file")?;
        Ok(config)
    }
}
