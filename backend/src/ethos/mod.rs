//! Effect Ethos: Deontic Guardrails for AI Safety
//!
//! This module implements "Compliance Guardrails" that block unsafe predictions
//! and provide counterfactual explanations for why actions were blocked.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Explanation generated when an action is blocked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualExplanation {
    /// The action that was attempted
    pub blocked_action: String,
    /// The rule that was violated
    pub rule_violated: String,
    /// The rule ID for audit logging
    pub rule_id: String,
    /// What would need to change for the action to proceed
    pub counterfactual: String,
    /// Severity level (1-10)
    pub severity: u8,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl CounterfactualExplanation {
    pub fn new(
        blocked_action: impl Into<String>,
        rule_violated: impl Into<String>,
        rule_id: impl Into<String>,
        counterfactual: impl Into<String>,
        severity: u8,
    ) -> Self {
        Self {
            blocked_action: blocked_action.into(),
            rule_violated: rule_violated.into(),
            rule_id: rule_id.into(),
            counterfactual: counterfactual.into(),
            severity,
            context: HashMap::new(),
        }
    }

    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// Result of an ethos check
#[derive(Debug)]
pub enum EthosResult<T> {
    /// Action is allowed, proceed with the contained value
    Allowed(T),
    /// Action is blocked, explanation provided
    Blocked(CounterfactualExplanation),
}

impl<T> EthosResult<T> {
    pub fn is_allowed(&self) -> bool {
        matches!(self, EthosResult::Allowed(_))
    }

    pub fn is_blocked(&self) -> bool {
        matches!(self, EthosResult::Blocked(_))
    }

    pub fn unwrap(self) -> T {
        match self {
            EthosResult::Allowed(v) => v,
            EthosResult::Blocked(e) => panic!("Action blocked: {}", e.rule_violated),
        }
    }

    pub fn explanation(&self) -> Option<&CounterfactualExplanation> {
        match self {
            EthosResult::Blocked(e) => Some(e),
            _ => None,
        }
    }
}

/// Trait for defining ethos rules
pub trait EthosRule: Send + Sync {
    /// Unique identifier for this rule
    fn id(&self) -> &str;
    
    /// Human-readable description
    fn description(&self) -> &str;
    
    /// Check if the rule is satisfied given the patient data
    fn check(&self, data: &PatientData) -> bool;
    
    /// Generate counterfactual explanation when rule is violated
    fn explain(&self, data: &PatientData) -> CounterfactualExplanation;
}

/// Patient data context for rule evaluation
#[derive(Debug, Clone, Default)]
pub struct PatientData {
    pub vitals: HashMap<String, Option<f64>>,
    pub lab_values: HashMap<String, Option<f64>>,
    pub timestamps: HashMap<String, i64>,
    pub metadata: HashMap<String, String>,
}

impl PatientData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_vital(&mut self, name: impl Into<String>, value: Option<f64>) {
        self.vitals.insert(name.into(), value);
    }

    pub fn set_lab(&mut self, name: impl Into<String>, value: Option<f64>) {
        self.lab_values.insert(name.into(), value);
    }

    pub fn get_vital(&self, name: &str) -> Option<f64> {
        self.vitals.get(name).copied().flatten()
    }

    pub fn get_lab(&self, name: &str) -> Option<f64> {
        self.lab_values.get(name).copied().flatten()
    }

    pub fn is_vital_missing(&self, name: &str) -> bool {
        self.vitals.get(name).map_or(true, |v| v.is_none())
    }

    pub fn is_lab_missing(&self, name: &str) -> bool {
        self.lab_values.get(name).map_or(true, |v| v.is_none())
    }
}

/// Rule: Require critical vitals before prediction
pub struct RequireCriticalVitals {
    required_vitals: Vec<String>,
}

impl RequireCriticalVitals {
    pub fn new(vitals: Vec<&str>) -> Self {
        Self {
            required_vitals: vitals.into_iter().map(String::from).collect(),
        }
    }
}

impl EthosRule for RequireCriticalVitals {
    fn id(&self) -> &str {
        "ETHOS-001"
    }

    fn description(&self) -> &str {
        "Require critical vital signs before making predictions"
    }

    fn check(&self, data: &PatientData) -> bool {
        self.required_vitals.iter().all(|v| !data.is_vital_missing(v))
    }

    fn explain(&self, data: &PatientData) -> CounterfactualExplanation {
        let missing: Vec<_> = self.required_vitals
            .iter()
            .filter(|v| data.is_vital_missing(v))
            .cloned()
            .collect();

        CounterfactualExplanation::new(
            "Sepsis Risk Prediction",
            format!("Missing critical vital signs: {:?}", missing),
            self.id(),
            format!("If {} were available, prediction would proceed", missing.join(", ")),
            8,
        )
    }
}

/// Rule: Block prediction if uncertainty is too high
pub struct MaxUncertaintyThreshold {
    threshold: f64,
}

impl MaxUncertaintyThreshold {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl EthosRule for MaxUncertaintyThreshold {
    fn id(&self) -> &str {
        "ETHOS-002"
    }

    fn description(&self) -> &str {
        "Block prediction if data uncertainty exceeds threshold"
    }

    fn check(&self, data: &PatientData) -> bool {
        // Calculate uncertainty as percentage of missing values
        let total = data.vitals.len() + data.lab_values.len();
        if total == 0 {
            return false; // No data = high uncertainty
        }
        
        let missing = data.vitals.values().filter(|v| v.is_none()).count()
            + data.lab_values.values().filter(|v| v.is_none()).count();
        
        let uncertainty = missing as f64 / total as f64;
        uncertainty <= self.threshold
    }

    fn explain(&self, data: &PatientData) -> CounterfactualExplanation {
        let total = data.vitals.len() + data.lab_values.len();
        let missing = data.vitals.values().filter(|v| v.is_none()).count()
            + data.lab_values.values().filter(|v| v.is_none()).count();
        let uncertainty = if total > 0 { missing as f64 / total as f64 } else { 1.0 };

        CounterfactualExplanation::new(
            "Sepsis Risk Prediction",
            format!("Data uncertainty ({:.1}%) exceeds maximum threshold ({:.1}%)", 
                    uncertainty * 100.0, self.threshold * 100.0),
            self.id(),
            format!("If at least {:.0}% of values were present, prediction would proceed",
                    (1.0 - self.threshold) * 100.0),
            7,
        )
        .with_context("current_uncertainty", format!("{:.2}", uncertainty))
        .with_context("threshold", format!("{:.2}", self.threshold))
    }
}

/// Main Ethos Guard that checks all rules
pub struct EthosGuard {
    rules: Vec<Box<dyn EthosRule>>,
}

impl EthosGuard {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create a guard with default clinical rules
    pub fn clinical_default() -> Self {
        let mut guard = Self::new();
        
        // Require MAP and Heart Rate at minimum
        guard.add_rule(Box::new(RequireCriticalVitals::new(vec!["MAP", "HR"])));
        
        // Block if more than 50% of data is missing
        guard.add_rule(Box::new(MaxUncertaintyThreshold::new(0.5)));
        
        guard
    }

    pub fn add_rule(&mut self, rule: Box<dyn EthosRule>) {
        self.rules.push(rule);
    }

    /// Check all rules and return the first violation if any
    pub fn check<T>(&self, data: &PatientData, action: T) -> EthosResult<T> {
        for rule in &self.rules {
            if !rule.check(data) {
                return EthosResult::Blocked(rule.explain(data));
            }
        }
        EthosResult::Allowed(action)
    }

    /// Check all rules and collect ALL violations
    pub fn check_all(&self, data: &PatientData) -> Vec<CounterfactualExplanation> {
        self.rules
            .iter()
            .filter(|rule| !rule.check(data))
            .map(|rule| rule.explain(data))
            .collect()
    }
}

impl Default for EthosGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ethos_blocks_missing_vitals() {
        let guard = EthosGuard::clinical_default();
        let mut data = PatientData::new();
        
        // Missing MAP and HR should be blocked
        let result = guard.check(&data, "prediction");
        assert!(result.is_blocked());
        
        // Add MAP but not HR - still blocked
        data.set_vital("MAP", Some(75.0));
        let result = guard.check(&data, "prediction");
        assert!(result.is_blocked());
        
        // Add HR - now allowed
        data.set_vital("HR", Some(80.0));
        let result = guard.check(&data, "prediction");
        assert!(result.is_allowed());
    }

    #[test]
    fn test_counterfactual_explanation() {
        let rule = RequireCriticalVitals::new(vec!["MAP", "HR", "SpO2"]);
        let mut data = PatientData::new();
        data.set_vital("MAP", Some(75.0));
        
        let explanation = rule.explain(&data);
        assert!(explanation.counterfactual.contains("HR"));
        assert!(explanation.counterfactual.contains("SpO2"));
    }
}
