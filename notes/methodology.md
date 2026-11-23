# Methodology: Causal Drivers and Safety Guardrails for ICU Sepsis Early Warning

## 1. Purpose

Sepsis early-warning models are usually judged on one number, discrimination. This project asks two further questions that matter when a model's output changes what clinicians do:

1. **Explanation**: can a short list of physiologically plausible drivers, selected with causal-discovery tools rather than by the model, warn of sepsis hours before onset, and what does the short list cost in detection?
2. **Abstention**: can explicit, auditable rules stop the model from answering when the record is too sparse to trust, and do the hours they block really contain worse predictions?

Every number quoted in the README comes from `results/`, produced by the pipeline below.

## 2. Data

- **Source**: PhysioNet/Computing in Cardiology Challenge 2019 (Reyna et al., 2020): 40,336 ICU stays from two hospital systems, 1,552,210 patient-hours, 8 vital signs, 26 laboratory values and 6 demographic/administrative fields per hour.
- **Label**: the challenge's hourly `SepsisLabel`, based on Sepsis-3 and shifted so it is positive from 6 hours before clinical onset. Predicting the hourly label is therefore 6-hour early warning. 1.8% of hours and 7.3% of stays are positive.
- **Split**: hospital A (patient IDs below 100,000; 20,336 stays) trains everything, including feature selection and the alert threshold; hospital B (20,000 stays) is only used for evaluation. This is an external validation: different site, different case mix (8.8% vs 5.7% septic stays).
- **Missingness**: most labs are measured every few hours. Within each stay, every variable is carried forward from its last measurement, so the features at hour *t* use only information available at hour *t*. A flag records whether each value was actually measured that hour (used by the guardrails). `EtCO2` is dropped (stored as text, almost never present).

## 3. Causal discovery (Rust, hospital A only)

### 3.1 Candidate sets

Two mRMR runs over hospital A's 790,215 patient-hours:

- **Diagnostic run**: all physiological, demographic, ICU-time and administrative fields.
- **Causal-driver run**: physiological and demographic fields only. ICU length of stay (`ICULOS`), hospital-to-ICU time and unit flags are excluded by domain prior: they describe the care process and the label's construction, not mechanisms that cause sepsis.

### 3.2 mRMR

DeepCausality's `mrmr_features_selector` (FCQ variant: F-statistic relevance, correlation redundancy, quotient score; pairwise deletion of missing values) selects ten features from each candidate set. Results on hospital A:

- Diagnostic run: ICULOS ranks first, followed by HR, Unit1, HospAdmTime, Resp, Temp, BUN, Unit2, Hgb, WBC.
- Causal-driver run: Resp, HCO3, Temp, Hgb, HR, BUN, WBC, Fibrinogen, Glucose, Creatinine.

### 3.3 SURD

SURD (Martínez-Sánchez, Arranz & Lozano-Durán, 2024) splits the information a set of sources carries about a target into **unique** (only one source has it), **redundant** (several sources share it) and **synergistic** (only appears in combination) parts, plus a **leak**: the share of target uncertainty no source explains.

- **Sources**: the top five causal drivers (Resp, HCO3, Temp, Hgb, HR), each binned into tertiles on hospital A plus a fourth "not yet measured" state, because missingness is informative in ICU data.
- **Target**: the hourly label. The joint distribution is the 2 × 4⁵ histogram over hospital A's patient-hours.
- **Result**: label entropy 0.1508 bits; the five drivers jointly carry 0.0061 bits (4%, leak fraction 0.96). Of that, 0.0045 bits is synergistic, 0.0015 redundant and 0.0001 unique (temperature only). The three parts sum to the joint mutual information, as they should.
- **Diagram**: `results/surd_decomposition.svg` (scripts/plot_surd.py; lines are undirected because SURD measures information structure, not causal direction).
- **Interpretation**: a single hour's snapshot of these signs says little about whether sepsis begins within six hours, and what it does say comes from combinations. This motivates a non-linear detector over several drivers instead of single-variable rules.

### 3.4 Implementation correctness

Three defects in the original code were fixed, each with a test that would have caught it:

- **Tensor layout**: DataFrames were written column by column into a tensor declared `[rows, columns]`, so mRMR ranked scrambled data. `test_row_major_layout_lets_mrmr_find_the_signal` checks that mRMR finds a planted signal.
- **SURD input**: the raw data matrix was passed where SURD expects a joint probability table. `run_surd` now builds the binned histogram.
- **SURD computation**: DeepCausality's `surd_states` (commit 6cc7f6f, code unchanged on current `main`) returns wrong values when the target distribution is not uniform (for a single source with p(T) = (0.7, 0.3) it reports 0.110 bits where the mutual information is 0.192), files unique information as redundant when another source carries zero specific information, fails with `ShapeMismatch` when target and source state counts differ, and panics when a source state never occurs. The same defects are reported upstream for release 0.5.4 in [deep_causality#878](https://github.com/deepcausality-rs/deep_causality/issues/878); they reproduce at 6cc7f6f as well. `surd_aggregate.rs` ports the aggregate decomposition without these defects. It reproduces exact cases (XOR → 1 bit synergy; duplicated source → 1 bit redundancy; driver plus independent noise → 1 bit unique to the driver) and agrees with a numpy transcription of the reference algorithm (`scripts/surd_reference.py`) to within 5e-13 bits.

The original "dual SURD" design (separate analyses of septic and non-septic hours) was dropped: within each subset the label is constant, so there is no information about it to decompose.

## 4. Detector

- **Model**: LightGBM (learning rate 0.05, 63 leaves, at least 200 rows per leaf, feature and row subsampling 0.8), early-stopped on a held-out 15% of hospital A's patients.
- **Features**: the ten causal drivers, as last-known values.
- **Baselines**, all evaluated identically on hospital B:
  - ICU length of stay alone (tests the label artifact);
  - a SIRS + partial qSOFA count (no learning; mentation is not in the data);
  - LightGBM on the diagnostic mRMR set, on all features, and on all features without time/administrative fields.
- **Device**: CPU. On the Intel Arc iGPU, LightGBM's OpenCL learner took 23.0 s against 11.8 s on CPU for 300 rounds on 670k × 40, so GPU training was not used.

## 5. Guardrails (Effect Ethos)

Two rules, implemented and unit-tested in Rust (`backend/src/ethos`) and applied identically in the Python evaluation:

- **ETHOS-001**: block if MAP or heart rate is missing.
- **ETHOS-002**: block if more than 50% of the 33 vitals and labs are missing.

"Missing" has two readings on hourly data, and both are reported:

- **Last known**: a value counts as present if it has been measured at any earlier point in the stay. This is the state a clinician sees.
- **This hour**: only values measured in this hour count.

Each prediction is reported with its guardrail verdict (and the rule that blocked it) and the five largest SHAP contributions (`results/example_reports.md`).

## 6. Evaluation

- **Discrimination**: hourly AUROC and AUPRC (hourly prevalence 1.4% on hospital B), and patient-level AUROC using each stay's maximum score. 95% confidence intervals come from 200 bootstrap resamples of whole patients, so correlated hours within a stay are not treated as independent.
- **Alerting**: an alert is the first hour a score crosses a threshold. The threshold is set on hospital A validation patients so that 80% of never-septic patients never alert. On hospital B we report sensitivity before onset (onset = first labelled hour + 6), specificity, PPV and lead time.
- **Guardrails**: coverage (hours allowed), AUROC on allowed vs blocked hours, and alerting with blocked hours unable to alert.

## 7. Results

| Detector | Hourly AUROC (95% CI) | Hourly AUPRC | Patient AUROC |
| --- | --- | --- | --- |
| ICU length of stay only | 0.628 (0.610–0.644) | 0.048 | 0.519 |
| SIRS + qSOFA clinical score | 0.661 (0.650–0.674) | 0.028 | 0.670 |
| Causal drivers (10 features) | 0.696 (0.682–0.707) | 0.039 | 0.722 |
| mRMR incl. time/admin fields | 0.749 (0.735–0.765) | 0.068 | 0.737 |
| All features, no time/admin | 0.753 (0.742–0.763) | 0.049 | 0.774 |
| All features | 0.768 (0.754–0.782) | 0.065 | 0.763 |

| Alerting on hospital B | Sensitivity before onset | Specificity | PPV | Median lead (IQR), hours |
| --- | --- | --- | --- | --- |
| Causal drivers | 0.340 | 0.896 | 0.178 | 28 (5–71) |
| Causal drivers + guardrails (last known) | 0.325 | 0.908 | 0.191 | 28 (5–73) |
| All features | 0.475 | 0.892 | 0.220 | 39 (13–86) |
| All features, no time/admin | 0.401 | 0.892 | 0.199 | 33 (5–80) |
| SIRS + qSOFA (score ≥ 4) | 0.558 | 0.689 | 0.103 | 29 (9–75) |

| Guardrail reading | Hours allowed | Labelled-sepsis hours allowed | AUROC allowed | AUROC blocked |
| --- | --- | --- | --- | --- |
| Last known | 69.9% | 83.8% | 0.708 | 0.627 |
| This hour | 3.6% | 6.1% | 0.711 | 0.695 |

## 8. Findings

1. **Explanation has a price.** The ten causal drivers beat the clinical score (0.696 vs 0.661 hourly AUROC) but trail the full feature set by about 0.07. The short list is easier to explain and audit; it is not free.
2. **The strongest "driver" was an artifact.** ICU length of stay predicts the label hour (0.628) but not the patient (0.519), because labels sit late in long stays. It tops the unconstrained mRMR ranking, and models that include it gain hourly AUROC without gaining patient-level discrimination (all features: 0.768 hourly, 0.763 patient; without time/admin: 0.753 hourly, 0.774 patient).
3. **Guardrails filter the right hours, if read sensibly.** With last-known values they block 30% of hours, on which the model is weaker on average (AUROC 0.627 vs 0.708 on allowed hours), and 16% of labelled-sepsis hours. They do not single out the worst predictions. Read literally on hourly data, ETHOS-002 blocks 96% of hours: rule design has to account for how often things are measured.
4. **Early is not the same as specific.** Median lead times of 28–39 hours with wide spreads mean many "early" alerts are probably raised by general illness, not by sepsis that is about to begin. Lead time should be read together with PPV (0.18–0.22).
5. **Operating points move between hospitals.** A threshold tuned for 80% specificity on hospital A gave about 90% on hospital B.

## 9. Assumptions and limitations

- **Selection is not causation.** "Causal drivers" means selected under a domain prior and an information criterion. Confounding by care is likely: clinicians order lactate or cultures when they already suspect sepsis, so measurement carries information that is not physiology.
- **SURD's causal reading** assumes the sources precede the target and capture the relevant state; with hourly snapshots and a label that is itself shifted in time, the decomposition is best read as information structure, not mechanism.
- One dataset with two sites and one label definition; no prospective or clinician evaluation.
- Binning (tertiles + missing) and the choice of five SURD sources affect the decomposition; they were fixed in advance in `config/default.toml`, not tuned on results.

## 10. Not implemented

The earlier version of this document described components that were never built: per-patient context hypergraphs (`backend/src/context` is a placeholder), a diagnostic "rapport" format scored by clinicians, and MIMIC as a data source. They are listed here so that nothing in the repository claims more than the code does.

## References

1. Reyna, M. A., et al. (2020). Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210–217.
2. Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801–810.
3. Martínez-Sánchez, Á., Arranz, G., & Lozano-Durán, A. (2024). Decomposing causality into its synergistic, unique, and redundant components. *Nature Communications*, 15.
4. Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. *IEEE TPAMI*, 27(8), 1226–1238.
5. DeepCausality: <https://github.com/deepcausality-rs/deep_causality>
