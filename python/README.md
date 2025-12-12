# Deep Causality Python Bindings

Python bindings for the Deep Causality Rust causal inference engine.

## Installation

```bash
# From source (requires Rust toolchain)
pip install maturin
cd python && maturin develop

# Or install the wheel directly
pip install deep_causality
```

## Quick Start

```python
import deep_causality
import polars as pl

# Load your data
df = pl.read_parquet("data/icu_sepsis.parquet")

# Run mRMR feature selection
features = deep_causality.run_mrmr_polars(
    df, 
    target="SepsisLabel", 
    max_features=10
)

# Print top features
for f in features:
    print(f"{f.name}: {f.score:.4f}")
```

## API Reference

### `run_mrmr_polars(df, target, max_features=10)`
Run mRMR on a Polars DataFrame.

### `run_mrmr(data, column_names, target, max_features=10)`
Run mRMR on raw 2D list data.

### `FeatureRanking`
Result object with `.name` (str) and `.score` (float) attributes.
