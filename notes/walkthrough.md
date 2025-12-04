# Project Walkthrough

## Quick Start

1.  **Setup Environment**:
    *Installs Python dependencies and prepares the system.*
    ```bash
    ./scripts/setup_with_delay.sh
    ```

2.  **Run Experiment**:
    *Runs the Rust backend to process data and find causal links.*
    ```bash
    cd backend
    cargo run --release
    ```

3.  **Analyze Results**:
    *Opens interactive notebooks to visualize the data and findings.*
    Open the notebooks in `notebooks/` using Marimo:
    ```bash
    source .venv/bin/activate
    # EDA (Exploratory Data Analysis)
    marimo edit notebooks/eda.py
    # Results Visualization
    marimo edit notebooks/results_analysis.py
    ```

## Project Structure
-   `backend/`: Rust causal inference engine.
-   `notebooks/`: Python analysis tools.
-   `data/`: PhysioNet dataset.
-   `scripts/`: Setup and verification scripts.
