# Deep Causality Methodology: Context-First Approach

## Core Philosophy
Traditional machine learning seeks statistical correlations that separate classes. In highly imbalanced data (like sepsis), this fails because the "signal" of the minority class is drowned out by the "noise" of the majority.

**DeepCausality** flips this approach:
1.  **Mechanism over Correlation**: We seek the *causal mechanism* of sepsis, not just a decision boundary.
2.  **Context-First**: A patient is not a vector; they are a dynamic context. The model must adapt to the individual's state.

## The Algorithm: SURD + mRMR
We employ a multi-stage causal discovery process:

### 1. Targeted Discovery (Dual Analysis)
We run the **SURD** (Synergistic Unique Redundant Degree) algorithm on two separate subsets:
-   **Subset A (Sepsis)**: What causes sepsis to manifest?
-   **Subset B (Non-Sepsis)**: What characterizes the "sick but not septic" state?

**Why?**
-   **Overlapping Features**: Identifying `Redundant` features across both sets helps filter out general ICU "noise" (general sickness).
-   **Disjoint Dominant Features**: Identifying `Unique` drivers in Subset A that are absent in Subset B reveals the *true* biomarkers of sepsis.

### 2. Dynamic Context Hypergraphs
Instead of a static model, we build a **CausaloidGraph**:
-   **Nodes (Causaloids)**: Represent causal functions (e.g., "If Lactate rises > 2.0 AND BP drops...").
-   **Edges**: Represent causal influence.
-   **Context**: A dynamic store of patient data, indexed relatively (e.g., "Time since admission").

### 3. The "Diagnostic Rapport"
The ultimate output is not just a probability, but a verifiable report:
-   **Traceability**: "Sepsis Risk High because [Lactate] triggered [Rule X]."
-   **Uncertainty**: "Confidence is 75% because [O2Sat] data is missing."
-   **Actionability**: Provides clinicians with the *why* behind the prediction.
