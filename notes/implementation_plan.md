# Deep Causality ICU Sepsis Implementation Plan

## Goal Description
Implement a high-performance, production-ready causal inference system for early sepsis detection using the DeepCausality library. The project aims to demonstrate "Context-First Causality" to handle highly imbalanced ICU data (7.27% sepsis prevalence) without relying on statistical correlations.

## Architecture
- **Backend (Rust)**: Core logic, data processing, causal discovery (SURD/mRMR), and inference engine. Prioritizes memory safety, speed, and concurrency.
- **Frontend (Python 3.12)**: Data analysis, visualization, and experiment tracking using Marimo notebooks.
- **Data**: PhysioNet Challenge 2019 dataset (ICU Sepsis).

## Proposed Changes

### 1. Project Structure & Configuration
- **Directory**: `Deep_Causality`
- **Config**: TOML-based configuration for experiment parameters (paths, thresholds, feature selection settings).
- **Logging**: Structured logging for experiment tracking.

### 2. Backend Implementation (Rust)
- **Data Layer**:
    - Use `Polars` for fast parquet/csv reading and manipulation.
    - Implement strict data validation (schema checks, missing value handling).
    - `MaybeUncertain<T>` pattern for handling missing ICU data.
- **Causal Discovery**:
    - Implement mRMR (Minimum Redundancy Maximum Relevance) for feature selection.
    - Implement SURD (Synergistic Unique Redundant Degree) for causal mechanism identification.
    - Dual analysis: Run discovery on "Sepsis" vs "Non-Sepsis" subsets to identify disjoint dominant features.
- **Context Engine**:
    - Build Dynamic Context Hypergraphs per patient.
    - Implement relative indexing (time relative to admission/onset).
    - CausaloidGraph construction based on discovered causal drivers.

### 3. Frontend Implementation (Python)
- **Environment**: Python 3.12 with `marimo`, `polars`, `matplotlib`/`seaborn`.
- **Notebooks**:
    - `eda.py`: Exploratory Data Analysis.
    - `results_analysis.py`: Visualization of SURD results and model performance.
    - `diagnostic_rapport.py`: Prototype of the clinician-facing report.

### 4. Verification Plan
- **Automated Tests**:
    - Unit tests for all Rust modules (data loading, algorithms).
    - Integration tests for the full pipeline.
- **Validation**:
    - Split data: 80% Train, 20% Validation (stratified).
    - Metrics: Precision, Recall (crucial for imbalance), F1-score, Clinical Utility Score.
    - Compare "Sepsis" vs "Non-Sepsis" causal drivers to validate the "Context-First" hypothesis.
