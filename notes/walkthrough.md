# Project Walkthrough

## Quick Start

1.  **Setup Environment**:
    ```bash
    ./scripts/setup_with_delay.sh
    ```

2.  **Run Experiment**:
    ```bash
    cd backend
    cargo run --release
    ```

3.  **Analyze Results**:
    Open the notebooks in `notebooks/` using Marimo:
    ```bash
    source .venv/bin/activate
    marimo edit notebooks/eda.py
    ```

## Project Structure
-   `backend/`: Rust causal inference engine.
-   `notebooks/`: Python analysis tools.
-   `data/`: PhysioNet dataset.
-   `scripts/`: Setup and verification scripts.
