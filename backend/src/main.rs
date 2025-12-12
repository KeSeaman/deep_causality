mod config;
mod data;
mod causality;
mod context;
mod utils;
mod ethos;
mod visualization;
mod inference;

use anyhow::Result;
use clap::Parser;
use tracing::{info, error, warn};
use crate::config::Config;
use crate::data::DataLoader;
use crate::causality::CausalDiscovery;
use crate::visualization::CausalGraph;
use crate::inference::{StreamingInference, StreamingConfig, VitalUpdate};

#[derive(Parser, Debug)]
#[command(author, version, about = "Deep Causality ICU Sepsis Causal Discovery Engine")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "../config/default.toml")]
    config: String,

    /// Run SURD dual analysis (Sepsis vs Non-Sepsis)
    #[arg(long, default_value = "false")]
    surd_analysis: bool,

    /// Export causal graph to DOT file
    #[arg(long)]
    export_graph: Option<String>,

    /// Run in real-time inference mode (reads JSON lines from stdin)
    #[arg(long, default_value = "false")]
    realtime: bool,

    /// Export results to JSON file
    #[arg(long)]
    export_json: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    
    info!("========================================");
    info!("  Deep Causality ICU Sepsis Backend");
    info!("========================================");
    
    let config = Config::load(&args.config)?;

    // Check for real-time mode first
    if args.realtime {
        return run_realtime_mode(&config).await;
    }

    // 1. Load Main Dataset
    info!("Loading training data from {}", config.data.train_path);
    match DataLoader::load_parquet(&config.data.train_path) {
        Ok(df) => {
            info!("Data loaded successfully. Shape: {:?}", df.shape());
            
            // 2. Run mRMR Feature Selection
            info!("\n--- mRMR Feature Selection ---");
            let features = match CausalDiscovery::run_mrmr(&df, &config.experiment.target_column, config.causality.max_features) {
                Ok(features) => {
                    info!("Top {} Selected Features:", features.len());
                    for (i, (name, score)) in features.iter().enumerate() {
                        info!("  {}. {} (score: {:.4})", i + 1, name, score);
                    }
                    features
                },
                Err(e) => {
                    error!("mRMR Feature Selection failed: {}", e);
                    vec![]
                }
            };

            // 3. Export causal graph if requested
            if let Some(graph_path) = &args.export_graph {
                info!("\n--- Exporting Causal Graph ---");
                let graph = CausalGraph::from_mrmr_results(&features, &config.experiment.target_column);
                graph.write_dot(graph_path)?;
                info!("Graph exported to {}", graph_path);
                
                // Also export JSON for web visualization
                let json_path = graph_path.replace(".dot", ".json");
                std::fs::write(&json_path, graph.to_json()?)?;
                info!("Graph JSON exported to {}", json_path);
            }

            // 4. Run SURD Dual Analysis if requested
            if args.surd_analysis {
                info!("\n--- SURD Dual Analysis ---");
                run_surd_dual_analysis(&config).await?;
            }
        },
        Err(e) => {
            error!("Failed to load data: {}. Please ensure data is present.", e);
        }
    }

    info!("\n========================================");
    info!("  Analysis Complete");
    info!("========================================");
    
    Ok(())
}

async fn run_surd_dual_analysis(config: &Config) -> Result<()> {
    // Load Sepsis subset
    info!("Loading Sepsis subset from {}", config.data.sepsis_subset_path);
    let sepsis_df = match DataLoader::load_parquet(&config.data.sepsis_subset_path) {
        Ok(df) => {
            info!("Sepsis subset loaded: {} rows", df.height());
            df
        },
        Err(e) => {
            error!("Failed to load Sepsis subset: {}", e);
            return Err(e);
        }
    };

    // Load Non-Sepsis subset
    info!("Loading Non-Sepsis subset from {}", config.data.non_sepsis_subset_path);
    let non_sepsis_df = match DataLoader::load_parquet(&config.data.non_sepsis_subset_path) {
        Ok(df) => {
            info!("Non-Sepsis subset loaded: {} rows", df.height());
            df
        },
        Err(e) => {
            error!("Failed to load Non-Sepsis subset: {}", e);
            return Err(e);
        }
    };

    // Run SURD Dual Analysis
    match CausalDiscovery::run_surd_dual(&sepsis_df, &non_sepsis_df, &config.experiment.target_column) {
        Ok(result) => {
            info!("\n=== SURD Dual Analysis Results ===\n");
            
            info!("SEPSIS Subset Information Decomposition:");
            info!("  Redundant (shared):     {:.4} bits", result.sepsis_result.redundant_info);
            info!("  Unique (discriminative): {:.4} bits", result.sepsis_result.unique_info);
            info!("  Synergistic (combined):  {:.4} bits", result.sepsis_result.synergistic_info);
            info!("  Total Information:       {:.4} bits", result.sepsis_result.total_info);
            
            info!("\nNON-SEPSIS Subset Information Decomposition:");
            info!("  Redundant (shared):     {:.4} bits", result.non_sepsis_result.redundant_info);
            info!("  Unique (discriminative): {:.4} bits", result.non_sepsis_result.unique_info);
            info!("  Synergistic (combined):  {:.4} bits", result.non_sepsis_result.synergistic_info);
            info!("  Total Information:       {:.4} bits", result.non_sepsis_result.total_info);
            
            info!("\n=== Causal Driver Comparison ===");
            info!("Sepsis-Specific Drivers (disjoint): {:?}", result.disjoint_drivers);
            info!("Shared Drivers (both groups): {:?}", result.shared_drivers);
            info!("Sepsis Specificity Score: {:.4}", result.sepsis_specific_score);

            // Export to JSON
            let json_output = serde_json::to_string_pretty(&result)?;
            std::fs::write("../notes/surd_results.json", &json_output)?;
            info!("\nResults exported to notes/surd_results.json");
        },
        Err(e) => {
            warn!("SURD Dual Analysis encountered an error: {}", e);
            warn!("Falling back to mRMR comparison.");
            run_mrmr_comparison(&sepsis_df, &non_sepsis_df, &config.experiment.target_column)?;
        }
    }

    Ok(())
}

fn run_mrmr_comparison(sepsis_df: &polars::prelude::DataFrame, non_sepsis_df: &polars::prelude::DataFrame, target_col: &str) -> Result<()> {
    info!("\n--- mRMR Feature Comparison (Sepsis vs Non-Sepsis) ---\n");
    
    let sepsis_features = CausalDiscovery::run_mrmr(sepsis_df, target_col, 10)?;
    let non_sepsis_features = CausalDiscovery::run_mrmr(non_sepsis_df, target_col, 10)?;

    info!("SEPSIS Top Features:");
    for (i, (name, score)) in sepsis_features.iter().enumerate() {
        info!("  {}. {} ({:.4})", i + 1, name, score);
    }

    info!("\nNON-SEPSIS Top Features:");
    for (i, (name, score)) in non_sepsis_features.iter().enumerate() {
        info!("  {}. {} ({:.4})", i + 1, name, score);
    }

    // Find differences
    let sep_names: std::collections::HashSet<_> = sepsis_features.iter().map(|(n, _)| n.clone()).collect();
    let non_sep_names: std::collections::HashSet<_> = non_sepsis_features.iter().map(|(n, _)| n.clone()).collect();
    
    let sepsis_only: Vec<_> = sep_names.difference(&non_sep_names).collect();
    let shared: Vec<_> = sep_names.intersection(&non_sep_names).collect();

    info!("\n=== Feature Comparison ===");
    info!("Sepsis-Only Features: {:?}", sepsis_only);
    info!("Shared Features: {:?}", shared);

    // Save comparison results
    let comparison = serde_json::json!({
        "sepsis_features": sepsis_features,
        "non_sepsis_features": non_sepsis_features,
        "sepsis_only": sepsis_only,
        "shared": shared,
    });
    std::fs::write("../notes/mrmr_comparison.json", serde_json::to_string_pretty(&comparison)?)?;
    info!("Comparison results saved to notes/mrmr_comparison.json");

    Ok(())
}

async fn run_realtime_mode(config: &Config) -> Result<()> {
    info!("\n--- Real-Time Inference Mode ---");
    info!("Reading JSON lines from stdin. Press Ctrl+C to stop.\n");

    // Load feature weights from mRMR on training data
    let feature_weights = match DataLoader::load_parquet(&config.data.train_path) {
        Ok(df) => {
            CausalDiscovery::run_mrmr(&df, &config.experiment.target_column, config.causality.max_features)
                .unwrap_or_default()
        },
        Err(_) => vec![
            ("ICULOS".to_string(), 1.0),
            ("HR".to_string(), 0.8),
            ("MAP".to_string(), 0.7),
            ("Lactate".to_string(), 0.6),
        ],
    };

    let streaming_config = StreamingConfig::default();
    let mut engine = StreamingInference::new(streaming_config);
    engine.set_feature_weights(feature_weights);

    // Read JSON lines from stdin
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let reader = stdin.lock();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<VitalUpdate>(&line) {
            Ok(update) => {
                let patient_id = update.patient_id.clone();
                match engine.process_update(update) {
                    Ok((result, alerts)) => {
                        if let Some(r) = result {
                            println!("{}", serde_json::to_string(&r)?);
                        }
                        for alert in alerts {
                            eprintln!("ALERT: {}", serde_json::to_string(&alert)?);
                        }
                    },
                    Err(e) => {
                        error!("Error processing update for {}: {}", patient_id, e);
                    }
                }
            },
            Err(e) => {
                warn!("Failed to parse JSON: {}", e);
            }
        }
    }

    Ok(())
}
