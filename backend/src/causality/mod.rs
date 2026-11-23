//! Causal discovery on the training hospital: DeepCausality's mRMR for feature selection and a
//! SURD information decomposition of what the top drivers say about the sepsis label.

pub mod surd_aggregate;

use crate::utils::tensor_adapter::TensorAdapter;
use anyhow::{Context, Result};
use deep_causality_algorithms::mrmr::mrmr_features_selector;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use surd_aggregate::surd_aggregate;
use tracing::info;

pub struct CausalDiscovery;

/// A feature subset and the information (bits) attributed to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoTerm {
    pub features: Vec<String>,
    pub bits: f64,
}

/// SURD decomposition of what a set of source features tells us about the target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurdSummary {
    pub sources: Vec<String>,
    pub bins_per_source: usize,
    pub rows: usize,
    pub target_entropy_bits: f64,
    /// Mutual information I(target; all sources).
    pub joint_mutual_info_bits: f64,
    /// Information only one source provides (SURD "unique").
    pub unique: Vec<InfoTerm>,
    /// Information several sources provide in common (SURD "redundant", order >= 2).
    pub redundant: Vec<InfoTerm>,
    /// Information that appears only when sources are combined.
    pub synergistic: Vec<InfoTerm>,
    pub total_unique_bits: f64,
    pub total_redundant_bits: f64,
    pub total_synergistic_bits: f64,
    /// H(target | all sources) / H(target): the share of label uncertainty the sources leave.
    pub info_leak_fraction: f64,
}

impl CausalDiscovery {
    /// mRMR (FCQ variant) over `candidates`, returning (feature, normalised score) in selection order.
    pub fn run_mrmr(df: &DataFrame, target_col: &str, candidates: &[String], k: usize) -> Result<Vec<(String, f64)>> {
        let mut cols: Vec<&str> = candidates.iter().map(String::as_str).collect();
        cols.push(target_col);
        let sub = df.select(cols).context("Selecting mRMR columns")?;
        let (tensor, names) = TensorAdapter::df_to_tensor(&sub)?;
        let target_idx = names.len() - 1;

        info!("mRMR over {} candidates, {} rows", candidates.len(), sub.height());
        let selected = mrmr_features_selector(&tensor, k, target_idx).map_err(|e| anyhow::anyhow!("mRMR failed: {:?}", e))?;
        Ok(selected.into_iter().map(|(i, s)| (names[i].clone(), s)).collect())
    }

    /// SURD on `sources` -> binary `target_col`.
    ///
    /// Each source is cut into `bins` quantile bins (edges from this DataFrame) plus one extra
    /// state for "not yet measured", because in ICU data missingness carries information.
    pub fn run_surd(df: &DataFrame, target_col: &str, sources: &[String], bins: usize) -> Result<SurdSummary> {
        let states = bins + 1;
        let target: Vec<usize> = df
            .column(target_col)?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| usize::from(v.unwrap_or(0.0) > 0.5))
            .collect();
        let binned: Vec<Vec<usize>> = sources.iter().map(|s| quantile_bin(df.column(s)?, bins)).collect::<Result<_>>()?;

        // Joint histogram, row-major over [target, source_1, ..., source_k].
        let shape: Vec<usize> = std::iter::once(2).chain(std::iter::repeat(states).take(sources.len())).collect();
        let mut counts = vec![0f64; shape.iter().product()];
        for row in 0..target.len() {
            let idx = binned.iter().fold(target[row], |acc, b| acc * states + b[row]);
            counts[idx] += 1.0;
        }
        info!("SURD on {} sources x {} states ({} cells, {} rows)", sources.len(), states, counts.len(), target.len());
        let r = surd_aggregate(&counts, &shape);

        // Sources are numbered from 1 (index 0 is the target).
        let terms = |m: &BTreeMap<Vec<usize>, f64>, keep: &dyn Fn(usize) -> bool| {
            let mut v: Vec<InfoTerm> = m
                .iter()
                .filter(|(k, _)| keep(k.len()))
                .map(|(k, &bits)| InfoTerm { features: k.iter().map(|&i| sources[i - 1].clone()).collect(), bits })
                .collect();
            v.sort_by(|a, b| b.bits.total_cmp(&a.bits));
            v
        };
        let unique = terms(&r.redundant, &|n| n == 1);
        let redundant = terms(&r.redundant, &|n| n > 1);
        let synergistic = terms(&r.synergistic, &|_| true);
        let total = |v: &[InfoTerm]| v.iter().map(|t| t.bits).sum::<f64>();
        let all: Vec<usize> = (1..=sources.len()).collect();

        Ok(SurdSummary {
            sources: sources.to_vec(),
            bins_per_source: bins,
            rows: target.len(),
            target_entropy_bits: r.target_entropy_bits,
            joint_mutual_info_bits: r.mutual_info.get(&all).copied().unwrap_or(0.0),
            total_unique_bits: total(&unique),
            total_redundant_bits: total(&redundant),
            total_synergistic_bits: total(&synergistic),
            unique,
            redundant,
            synergistic,
            info_leak_fraction: r.info_leak_fraction,
        })
    }
}

/// Quantile bins 0..bins-1 for observed values; `bins` for missing.
fn quantile_bin(col: &Series, bins: usize) -> Result<Vec<usize>> {
    let values: Vec<Option<f64>> = col.cast(&DataType::Float64)?.f64()?.into_iter().collect();
    let mut observed: Vec<f64> = values.iter().flatten().copied().filter(|v| v.is_finite()).collect();
    observed.sort_by(f64::total_cmp);
    let edges: Vec<f64> = (1..bins).map(|q| observed.get(q * observed.len() / bins).copied().unwrap_or(f64::INFINITY)).collect();
    Ok(values
        .iter()
        .map(|v| match v {
            Some(x) if x.is_finite() => edges.iter().filter(|&&e| *x >= e).count(),
            _ => bins,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_bin_with_missing_state() -> Result<()> {
        let s = Series::new("x", &[Some(1.0), Some(2.0), Some(3.0), Some(4.0), None, Some(5.0)]);
        let b = quantile_bin(&s, 2)?;
        assert_eq!(b[4], 2, "missing gets its own state");
        assert!(b[0] < b[5], "low and high values land in different bins");
        Ok(())
    }

    #[test]
    fn test_surd_finds_the_informative_source() -> Result<()> {
        // "driver" determines the label; "noise" does not.
        let n = 4000;
        let driver: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let noise: Vec<f64> = (0..n).map(|i| ((i * 7919) % 13) as f64).collect();
        let label: Vec<f64> = driver.iter().map(|&d| if d >= 2.0 { 1.0 } else { 0.0 }).collect();
        let df = df!["driver" => driver, "noise" => noise, "label" => label]?;

        let s = CausalDiscovery::run_surd(&df, "label", &["driver".into(), "noise".into()], 2)?;
        let bits = |f: &str| s.unique.iter().find(|t| t.features == vec![f.to_string()]).map_or(0.0, |t| t.bits);
        assert!(bits("driver") > 0.9, "driver should carry ~1 bit, got {}; {:?}", bits("driver"), s);
        assert!(bits("noise") < 0.05, "noise should carry ~0 bits, got {}", bits("noise"));
        Ok(())
    }
}
