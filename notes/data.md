# ICU Sepsis Data Overview

## Dataset Statistics
- **Source**: PhysioNet Computing in Cardiology Challenge 2019.
- **Population**: 40,336 unique ICU patients.
- **Records**: ~1.55 million total hourly records.
- **Density**: Average of 38 records per patient.
- **Target**: `SepsisLabel` (binary).

## The Imbalance Problem
- **Sepsis Prevalence**: Only 2,932 patients (7.27%) have sepsis.
- **Non-Sepsis**: 92.73% of patients are negative.
- **Implication**: Standard statistical models optimizing for accuracy will default to predicting "No Sepsis", achieving ~93% accuracy but missing all lethal cases. This is unacceptable in a clinical setting.

## Features
The dataset contains 40+ variables including:
- **Vitals**: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2.
- **Labs**: Lactate, WBC, Platelets, Creatinine, Bilirubin, etc.
- **Demographics**: Age, Gender, ICU Length of Stay (ICULOS).

## Data Quality & Handling
- **Missingness**: High frequency of missing values (common in ICU data).
- **Strategy** (scripts/prepare_data.py):
    - Rows are never dropped.
    - Within each ICU stay, every vital and lab is carried forward from its last measurement, so features at hour t use only information available at hour t.
    - A per-variable flag records whether the value was measured in that hour; the guardrails use it.
    - For SURD, each source gets an explicit "not yet measured" state alongside its tertile bins.
- **Expected location**: `data/all/dataset.parquet` (the challenge's per-patient files combined, with a `Patient_ID` column; IDs below 100,000 are hospital A, the rest hospital B).