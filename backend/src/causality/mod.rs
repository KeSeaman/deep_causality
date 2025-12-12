use crate::utils::tensor_adapter::TensorAdapter;
use deep_causality_algorithms::mrmr::mrmr_features_selector;
use deep_causality_algorithms::surd::{surd_states, SurdResult};
use polars::prelude::*;
use anyhow::{Result, Context};
use tracing::info;
use serde::{Serialize, Deserialize};

pub struct CausalDiscovery;

/// Result from SURD analysis containing decomposed causal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurdAnalysisResult {
    pub redundant_info: f64,
    pub unique_info: f64,
    pub synergistic_info: f64,
    pub total_info: f64,
}

/// Result from dual SURD analysis comparing Sepsis vs Non-Sepsis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurdDualResult {
    pub sepsis_result: SurdAnalysisResult,
    pub non_sepsis_result: SurdAnalysisResult,
    pub disjoint_drivers: Vec<String>,       // Unique to sepsis
    pub shared_drivers: Vec<String>,          // Present in both
    pub sepsis_specific_score: f64,           // Measure of how different sepsis drivers are
}

impl CausalDiscovery {
    /// Run mRMR feature selection algorithm
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

    /// Run SURD (Synergistic Unique Redundant Degree) analysis
    /// Returns decomposed information: Redundant, Unique, Synergistic
    pub fn run_surd(df: &DataFrame, target_col: &str) -> Result<SurdAnalysisResult> {
        info!("Converting DataFrame to CausalTensor for SURD...");
        let (tensor, col_names) = TensorAdapter::df_to_tensor(df)?;

        // Find target column index
        let target_idx = col_names.iter()
            .position(|n| n == target_col)
            .context(format!("Target column {} not found", target_col))?;

        // Get feature indices (all columns except target)
        let agent_indices: Vec<usize> = (0..col_names.len())
            .filter(|&i| i != target_idx)
            .collect();

        info!("Running SURD causal discovery with {} features...", agent_indices.len());
        
        // Call SURD algorithm
        let surd_result = surd_states(&tensor, target_idx, &agent_indices)
            .map_err(|e| anyhow::anyhow!("SURD execution failed: {:?}", e))?;

        // Aggregate SURD results
        let (redundant, unique, synergistic) = Self::aggregate_surd_result(&surd_result);
        let total = redundant + unique + synergistic;

        Ok(SurdAnalysisResult {
            redundant_info: redundant,
            unique_info: unique,
            synergistic_info: synergistic,
            total_info: total,
        })
    }

    /// Run dual SURD analysis: compare Sepsis vs Non-Sepsis subsets
    pub fn run_surd_dual(
        sepsis_df: &DataFrame, 
        non_sepsis_df: &DataFrame, 
        target_col: &str
    ) -> Result<SurdDualResult> {
        info!("=== SURD Dual Analysis: Sepsis vs Non-Sepsis ===");
        
        // Analyze Sepsis subset
        info!("Analyzing Sepsis subset ({} rows)...", sepsis_df.height());
        let sepsis_result = Self::run_surd(sepsis_df, target_col)?;
        
        // Analyze Non-Sepsis subset  
        info!("Analyzing Non-Sepsis subset ({} rows)...", non_sepsis_df.height());
        let non_sepsis_result = Self::run_surd(non_sepsis_df, target_col)?;

        // Run mRMR on both to identify feature rankings
        let sepsis_features = Self::run_mrmr(sepsis_df, target_col, 15)?;
        let non_sepsis_features = Self::run_mrmr(non_sepsis_df, target_col, 15)?;

        // Find disjoint (sepsis-only) and shared drivers
        let sepsis_names: std::collections::HashSet<_> = sepsis_features.iter()
            .map(|(name, _)| name.clone())
            .collect();
        let non_sepsis_names: std::collections::HashSet<_> = non_sepsis_features.iter()
            .map(|(name, _)| name.clone())
            .collect();

        let disjoint_drivers: Vec<String> = sepsis_names.difference(&non_sepsis_names)
            .cloned()
            .collect();
        let shared_drivers: Vec<String> = sepsis_names.intersection(&non_sepsis_names)
            .cloned()
            .collect();

        // Calculate sepsis specificity score (unique/total ratio difference)
        let sepsis_unique_ratio = if sepsis_result.total_info > 0.0 {
            sepsis_result.unique_info / sepsis_result.total_info
        } else { 0.0 };
        let non_sepsis_unique_ratio = if non_sepsis_result.total_info > 0.0 {
            non_sepsis_result.unique_info / non_sepsis_result.total_info
        } else { 0.0 };
        let sepsis_specific_score = (sepsis_unique_ratio - non_sepsis_unique_ratio).abs();

        Ok(SurdDualResult {
            sepsis_result,
            non_sepsis_result,
            disjoint_drivers,
            shared_drivers,
            sepsis_specific_score,
        })
    }

    /// Aggregate SURD result into (Redundant, Unique, Synergistic) totals
    fn aggregate_surd_result<T>(result: &SurdResult<T>) -> (f64, f64, f64) {
        let redundant: f64 = result.redundant_info().values().sum();
        let unique: f64 = result.mutual_info().values().sum(); // mutual_info represents unique contribution
        let synergistic: f64 = result.synergistic_info().values().sum();
        (redundant, unique, synergistic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surd_analysis_result_serialization() {
        let result = SurdAnalysisResult {
            redundant_info: 0.5,
            unique_info: 0.3,
            synergistic_info: 0.2,
            total_info: 1.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("redundant_info"));
    }
}
