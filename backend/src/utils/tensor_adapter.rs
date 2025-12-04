use polars::prelude::*;
use deep_causality_tensor::CausalTensor;
use anyhow::{Result, Context};

pub struct TensorAdapter;

impl TensorAdapter {
    pub fn df_to_tensor(df: &DataFrame) -> Result<(CausalTensor<Option<f64>>, Vec<String>)> {
        let (height, width) = df.shape();
        let mut flat_data: Vec<Option<f64>> = Vec::with_capacity(height * width);
        let mut column_names: Vec<String> = Vec::with_capacity(width);

        for col_name in df.get_column_names() {
            let series = df.column(col_name)?;
            column_names.push(col_name.to_string());

            // Cast to Float64 and handle nulls
            let ca = series.cast(&DataType::Float64)?;
            let f64_ca = ca.f64()?;

            for opt_val in f64_ca.into_iter() {
                flat_data.push(opt_val);
            }
        }

        // DeepCausality expects data in Column-Major order based on reference implementation
        let tensor = CausalTensor::new(flat_data, vec![height, width])
            .context("Failed to create CausalTensor")?;

        Ok((tensor, column_names))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_df_to_tensor() -> Result<()> {
        let df = df! [
            "a" => [1.0, 2.0],
            "b" => [3.0, 4.0]
        ]?;

        let (tensor, names) = TensorAdapter::df_to_tensor(&df)?;
        
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(tensor.shape(), &[2, 2]);
        
        Ok(())
    }
}
