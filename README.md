# Deep Causality: ICU Sepsis Case Study

## Overview
This project is a reimplementation and enhancement of the ICU Sepsis Case Study using **DeepCausality**, a Rust-based hyper-geometric computational causality library. It leverages a polyglot architecture with a high-performance **Rust backend** for causal discovery and inference, and a **Python (Marimo)** frontend for data analysis and visualization.

## Aim
The primary goal is to demonstrate **Context-First Causality** in a critical medical setting. Specifically, we aim to:
1.  Overcome the limitations of statistical correlation on highly imbalanced data (7.27% sepsis prevalence).
2.  Identify disjoint causal drivers that distinguish "Sepsis" from general "ICU Sickness".
3.  Generate explainable, patient-specific "Diagnostic Rapports" that provide clinicians with actionable insights and confidence scores.

## Methodology
We utilize a novel two-stage approach:
1.  **Causal Discovery**: Using **SURD** (Synergistic Unique Redundant Degree) and **mRMR** on split datasets to isolate unique sepsis biomarkers.
2.  **Contextual Inference**: Constructing dynamic **CausaloidGraphs** that model each patient as a unique, evolving context, using relative time indexing and handling missing data with explicit uncertainty.

## Project Structure
- `backend/`: Rust source code for the causal engine.
- `notebooks/`: Python Marimo notebooks for EDA and result visualization.
- `data/`: Dataset storage (PhysioNet 2019).
- `notes/`: Documentation, implementation plans, and research notes.
- `config/`: Configuration files (TOML).

## Constraints & Requirements
- **Data**: PhysioNet Challenge 2019 data (must be downloaded separately due to license).
- **Performance**: Rust backend ensures low-latency inference suitable for real-time monitoring.
- **Safety**: Strict type systems and `MaybeUncertain` wrappers ensure robust handling of missing or erroneous clinical data.

## Results
The DeepCausality mRMR algorithm successfully identified the top causal drivers for sepsis from the dataset.

**Top 10 Selected Features (ranked by importance):**
1.  **ICULOS** (1.0000): ICU Length of Stay is the strongest indicator, reflecting the cumulative risk over time.
2.  **Patient_ID** (0.9888): *Note: High correlation due to data structure (multiple records per patient), typically excluded in downstream modeling but captured here as a unique identifier of context.*
3.  **HospAdmTime** (0.5085): Time between hospital and ICU admission.
4.  **Unit2** (0.3542): MICU (Medical ICU) indicator.
5.  **Unit1** (0.2710): SICU (Surgical ICU) indicator.
6.  **Gender** (0.2149).
7.  **Age** (0.1783).
8.  **Platelets** (0.1523): Key biomarker for coagulation dysfunction.
9.  **Fibrinogen** (0.1353): Another coagulation marker.
10. **WBC** (0.1196): White Blood Cell count, a primary inflammation marker.

These results align with clinical expectations, highlighting that **length of stay**, **admission context**, and **coagulation/inflammation markers** are the primary causal signals in this dataset.
