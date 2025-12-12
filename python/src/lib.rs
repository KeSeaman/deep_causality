//! Python bindings for Deep Causality causal inference engine
//! 
//! Provides Python access to:
//! - mRMR feature selection
//! - SURD causal decomposition
//! - Causaloid graph construction

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use polars::prelude::*;
use anyhow::Result;
use deep_causality_algorithms::mrmr::mrmr_features_selector;
use deep_causality_tensor::CausalTensor;

/// Result from mRMR feature selection
#[pyclass]
#[derive(Clone)]
struct FeatureRanking {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    score: f64,
}

#[pymethods]
impl FeatureRanking {
    fn __repr__(&self) -> String {
        format!("FeatureRanking(name='{}', score={:.4})", self.name, self.score)
    }
}

/// Result from SURD analysis
#[pyclass]
#[derive(Clone)]
struct SurdResult {
    #[pyo3(get)]
    redundant_info: f64,
    #[pyo3(get)]
    unique_info: f64,
    #[pyo3(get)]
    synergistic_info: f64,
    #[pyo3(get)]
    total_info: f64,
}

#[pymethods]
impl SurdResult {
    fn __repr__(&self) -> String {
        format!(
            "SurdResult(redundant={:.4}, unique={:.4}, synergistic={:.4}, total={:.4})",
            self.redundant_info, self.unique_info, self.synergistic_info, self.total_info
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("redundant_info", self.redundant_info)?;
        dict.set_item("unique_info", self.unique_info)?;
        dict.set_item("synergistic_info", self.synergistic_info)?;
        dict.set_item("total_info", self.total_info)?;
        Ok(dict.into())
    }
}

/// Convert Python list of lists to CausalTensor
fn py_data_to_tensor(data: Vec<Vec<f64>>) -> Result<(CausalTensor<Option<f64>>, usize, usize), PyErr> {
    if data.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty data"));
    }
    
    let n_rows = data.len();
    let n_cols = data[0].len();
    
    // Flatten in column-major order (for compatibility with deep_causality)
    let mut flat_data: Vec<Option<f64>> = Vec::with_capacity(n_rows * n_cols);
    for col_idx in 0..n_cols {
        for row in &data {
            flat_data.push(Some(row[col_idx]));
        }
    }
    
    let tensor = CausalTensor::new(flat_data, vec![n_rows, n_cols])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
    
    Ok((tensor, n_rows, n_cols))
}

/// Run mRMR (Minimum Redundancy Maximum Relevance) feature selection
///
/// Args:
///     data: 2D list of floats (rows x columns)
///     column_names: List of column names
///     target_column: Name of the target column
///     max_features: Maximum number of features to select
///
/// Returns:
///     List of FeatureRanking objects, sorted by importance
#[pyfunction]
#[pyo3(signature = (data, column_names, target_column, max_features=10))]
fn run_mrmr(
    data: Vec<Vec<f64>>,
    column_names: Vec<String>,
    target_column: String,
    max_features: usize,
) -> PyResult<Vec<FeatureRanking>> {
    // Find target column index
    let target_idx = column_names.iter()
        .position(|n| n == &target_column)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Target column '{}' not found", target_column)
        ))?;

    // Convert to tensor
    let (tensor, _, _) = py_data_to_tensor(data)?;

    // Run mRMR
    let selected = mrmr_features_selector(&tensor, max_features, target_idx)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

    // Map back to names
    let results: Vec<FeatureRanking> = selected.into_iter()
        .map(|(idx, score)| FeatureRanking {
            name: column_names[idx].clone(),
            score,
        })
        .collect();

    Ok(results)
}

/// Run mRMR on a Polars DataFrame (passed as dict of columns)
///
/// Args:
///     df_dict: Dictionary mapping column names to lists of values
///     target_column: Name of the target column
///     max_features: Maximum number of features to select
///
/// Returns:
///     List of FeatureRanking objects
#[pyfunction]
#[pyo3(signature = (df_dict, target_column, max_features=10))]
fn run_mrmr_from_dict(
    py: Python,
    df_dict: &PyDict,
    target_column: String,
    max_features: usize,
) -> PyResult<Vec<FeatureRanking>> {
    let mut column_names: Vec<String> = Vec::new();
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut n_rows: Option<usize> = None;

    // Extract columns from dict
    for (key, value) in df_dict.iter() {
        let col_name: String = key.extract()?;
        let col_data: Vec<f64> = value.extract()?;
        
        if let Some(expected_rows) = n_rows {
            if col_data.len() != expected_rows {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "All columns must have the same length"
                ));
            }
        } else {
            n_rows = Some(col_data.len());
        }
        
        column_names.push(col_name);
        data.push(col_data);
    }

    // Transpose: from column-oriented to row-oriented
    let n_rows = n_rows.unwrap_or(0);
    let n_cols = column_names.len();
    let mut row_data: Vec<Vec<f64>> = vec![vec![0.0; n_cols]; n_rows];
    
    for (col_idx, col) in data.iter().enumerate() {
        for (row_idx, &val) in col.iter().enumerate() {
            row_data[row_idx][col_idx] = val;
        }
    }

    run_mrmr(row_data, column_names, target_column, max_features)
}

/// Get library version
#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

/// Main Python module
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FeatureRanking>()?;
    m.add_class::<SurdResult>()?;
    m.add_function(wrap_pyfunction!(run_mrmr, m)?)?;
    m.add_function(wrap_pyfunction!(run_mrmr_from_dict, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
