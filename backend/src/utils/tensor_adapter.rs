use anyhow::{Context, Result};
use deep_causality_tensor::CausalTensor;
use polars::prelude::*;

pub struct TensorAdapter;

impl TensorAdapter {
    /// Converts a DataFrame to a `[rows, cols]` tensor in row-major order (one sample per row),
    /// the layout DeepCausality's mRMR expects. Nulls become `None` (pairwise deletion in mRMR).
    pub fn df_to_tensor(df: &DataFrame) -> Result<(CausalTensor<Option<f64>>, Vec<String>)> {
        let (height, width) = df.shape();
        let column_names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let columns: Vec<Vec<Option<f64>>> = df
            .get_columns()
            .iter()
            .map(|s| -> Result<Vec<Option<f64>>> {
                let ca = s.cast(&DataType::Float64)?;
                Ok(ca.f64()?.into_iter().collect())
            })
            .collect::<Result<_>>()?;

        let flat_data: Vec<Option<f64>> = (0..height)
            .flat_map(|row| columns.iter().map(move |col| col[row]))
            .collect();

        let tensor = CausalTensor::new(flat_data, vec![height, width])
            .context("Failed to create CausalTensor")?;
        Ok((tensor, column_names))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deep_causality_algorithms::mrmr::mrmr_features_selector;

    #[test]
    fn test_df_to_tensor_shape() -> Result<()> {
        let df = df! [
            "a" => [1.0, 2.0],
            "b" => [3.0, 4.0]
        ]?;
        let (tensor, names) = TensorAdapter::df_to_tensor(&df)?;
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(tensor.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_row_major_layout_lets_mrmr_find_the_signal() -> Result<()> {
        // "signal" determines the target exactly; "noise" is unrelated. A column-major fill
        // would scramble rows and columns, and mRMR would not reliably pick "signal" first.
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let noise: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64).collect();
        let target: Vec<f64> = signal.iter().map(|s| 2.0 * s + 1.0).collect();
        let df = df! ["noise" => noise, "signal" => signal, "target" => target]?;

        let (tensor, names) = TensorAdapter::df_to_tensor(&df)?;
        let selected = mrmr_features_selector(&tensor, 1, 2).expect("mRMR failed");
        assert_eq!(names[selected[0].0], "signal");
        Ok(())
    }
}
