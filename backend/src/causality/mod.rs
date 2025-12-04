use crate::utils::tensor_adapter::TensorAdapter;
use deep_causality_algorithms::mrmr::mrmr_features_selector;
use polars::prelude::*;
use anyhow::{Result, Context};
use tracing::info;

pub struct CausalDiscovery;

impl CausalDiscovery {
    pub fn run_mrmr(df: &DataFrame, target_col: &str, max_features: usize) -> Result<Vec<(String, f64)>> {
        info!("Converting DataFrame to CausalTensor for mRMR...");
        let (tensor, col_names) = TensorAdapter::df_to_tensor(df)?;
        
        // Find target column index
        let target_idx = col_names.iter()
            .position(|n| n == target_col)
            .context(format!("Target column {} not found", target_col))?;

        info!("Running mRMR feature selection...");
        let selected_features = mrmr_features_selector(&tensor, max_features, target_idx)
            .map_err(|e| anyhow::anyhow!("mRMR execution failed: {:?}", e))?;

        // Map indices back to names
        let result: Vec<(String, f64)> = selected_features.into_iter()
            .map(|(idx, score)| (col_names[idx].clone(), score))
            .collect();

        Ok(result)
    }

    // Placeholder for SURD - assuming similar API or need to implement based on notes
    // The notes mention SURD is used for "Synergistic Unique Redundant Degree"
    // I'll assume there's a function for it or I need to look for it in the crate.
    // For now, I'll leave a placeholder or try to find it.
    pub fn run_surd(_df: &DataFrame) -> Result<()> {
        info!("SURD implementation pending...");
        Ok(())
    }
}
