#![allow(dead_code)]
mod causality;
mod config;
mod context;
mod data;
mod ethos;
mod utils;
mod visualization;

use crate::causality::{CausalDiscovery, SurdSummary};
use crate::config::Config;
use crate::data::DataLoader;
use crate::visualization::CausalGraph;
use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "Deep Causality: causal discovery for ICU sepsis (training hospital only)")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "../config/default.toml")]
    config: String,

    /// Also export the causal-driver graph as Graphviz DOT
    #[arg(long)]
    export_graph: Option<String>,
}

#[derive(Serialize)]
struct DiscoveryReport {
    train_rows: usize,
    train_patients: usize,
    /// Diagnostic: mRMR with time/administrative fields allowed.
    mrmr_with_time_admin: Vec<(String, f64)>,
    /// Causal-driver candidates: physiological and demographic fields only.
    mrmr_physiological: Vec<(String, f64)>,
    surd: SurdSummary,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let c = &config.causality;
    let target = &config.experiment.target_column;

    let df = DataLoader::load_parquet(&config.data.train_path)?;
    let train_patients = df.column(&config.experiment.patient_id_column)?.n_unique()?;
    info!("Training data: {} rows, {} patients", df.height(), train_patients);

    let with_time: Vec<String> = c.physiological.iter().chain(&c.time_admin).cloned().collect();
    let mrmr_with_time_admin = CausalDiscovery::run_mrmr(&df, target, &with_time, c.max_features)?;
    let mrmr_physiological = CausalDiscovery::run_mrmr(&df, target, &c.physiological, c.max_features)?;
    for (label, ranking) in [("with time/admin", &mrmr_with_time_admin), ("physiological", &mrmr_physiological)] {
        info!("mRMR {}:", label);
        for (i, (name, score)) in ranking.iter().enumerate() {
            info!("  {:>2}. {:<18} {:.4}", i + 1, name, score);
        }
    }

    let sources: Vec<String> = mrmr_physiological.iter().take(c.surd_sources).map(|(n, _)| n.clone()).collect();
    let surd = CausalDiscovery::run_surd(&df, target, &sources, c.surd_bins)?;
    info!(
        "SURD: H(target) {:.4} bits; unique {:.4}, redundant {:.4}, synergistic {:.4}, leak fraction {:.4}",
        surd.target_entropy_bits, surd.total_unique_bits, surd.total_redundant_bits, surd.total_synergistic_bits, surd.info_leak_fraction
    );

    if let Some(path) = &args.export_graph {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        CausalGraph::from_mrmr_results(&mrmr_physiological, target).write_dot(path)?;
        info!("Graph written to {}", path);
    }

    let report = DiscoveryReport { train_rows: df.height(), train_patients, mrmr_with_time_admin, mrmr_physiological, surd };
    if let Some(parent) = std::path::Path::new(&config.data.output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&config.data.output_path, serde_json::to_string_pretty(&report)?)?;
    info!("Results written to {}", config.data.output_path);
    Ok(())
}
