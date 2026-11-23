use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub data: DataConfig,
    pub experiment: ExperimentConfig,
    pub causality: CausalityConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DataConfig {
    pub train_path: String,
    pub output_path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExperimentConfig {
    pub target_column: String,
    pub patient_id_column: String,
    pub time_column: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CausalityConfig {
    pub physiological: Vec<String>,
    pub time_admin: Vec<String>,
    pub max_features: usize,
    pub surd_sources: usize,
    pub surd_bins: usize,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path).with_context(|| format!("Failed to read config file at {}", path))?;
        toml::from_str(&content).context("Failed to parse config file")
    }
}
