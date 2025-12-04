mod config;
mod data;
mod causality;
mod context;
mod utils;

use anyhow::Result;
use clap::Parser;
use tracing::{info, error};
use crate::config::Config;
use crate::data::DataLoader;
use crate::causality::CausalDiscovery;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "../config/default.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    
    info!("Starting Deep Causality ICU Sepsis Backend");
    let config = Config::load(&args.config)?;

    // 1. Load Data
    info!("Loading training data from {}", config.data.train_path);
    // Note: In a real run, we'd ensure this file exists or download it.
    // For now, we assume it's there or we handle the error gracefully.
    match DataLoader::load_parquet(&config.data.train_path) {
        Ok(df) => {
            info!("Data loaded successfully. Shape: {:?}", df.shape());
            
            // 2. Causal Discovery (mRMR)
            info!("Running Causal Discovery...");
            match CausalDiscovery::run_mrmr(&df, &config.experiment.target_column, config.causality.max_features) {
                Ok(features) => {
                    info!("Selected Features:");
                    for (name, score) in features {
                        info!("  {}: {:.4}", name, score);
                    }
                },
                Err(e) => error!("Causal Discovery failed: {}", e),
            }
        },
        Err(e) => {
            error!("Failed to load data: {}. Please ensure data is present.", e);
            // We don't exit here to allow the program to compile and run even without data for verification
        }
    }

    Ok(())
}
