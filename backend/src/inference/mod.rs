//! Real-Time Inference Module
//!
//! Provides streaming inference for continuous patient monitoring.
//! Processes patient data updates and triggers alerts when thresholds are exceeded.

use crate::ethos::{EthosGuard, EthosResult, PatientData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{info, warn, error};
use anyhow::Result;

/// Alert generated when a threshold is exceeded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub patient_id: String,
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: i64,
    pub triggering_values: HashMap<String, f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AlertType {
    SepsisRisk,
    VitalAbnormal,
    TrendChange,
    DataQuality,
    EthosBlocked,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Ord, PartialOrd, Eq)]
pub enum AlertSeverity {
    Info = 1,
    Warning = 2,
    Critical = 3,
    Emergency = 4,
}

/// Patient vital signs update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalUpdate {
    pub patient_id: String,
    pub timestamp: i64,
    pub vitals: HashMap<String, Option<f64>>,
    pub labs: HashMap<String, Option<f64>>,
}

impl VitalUpdate {
    pub fn to_patient_data(&self) -> PatientData {
        let mut data = PatientData::new();
        for (k, v) in &self.vitals {
            data.set_vital(k, *v);
        }
        for (k, v) in &self.labs {
            data.set_lab(k, *v);
        }
        data
    }
}

/// Inference result for a patient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub patient_id: String,
    pub timestamp: i64,
    pub sepsis_risk: f64,
    pub risk_level: RiskLevel,
    pub top_contributing_factors: Vec<(String, f64)>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Moderate,
    High,
    Critical,
}

impl RiskLevel {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s < 0.25 => RiskLevel::Low,
            s if s < 0.50 => RiskLevel::Moderate,
            s if s < 0.75 => RiskLevel::High,
            _ => RiskLevel::Critical,
        }
    }
}

/// Configuration for the streaming inference engine
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Alert threshold for sepsis risk (0.0-1.0)
    pub sepsis_alert_threshold: f64,
    /// Minimum time between alerts for same patient (seconds)
    pub alert_cooldown_secs: u64,
    /// Enable Ethos guardrails
    pub enable_ethos: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            sepsis_alert_threshold: 0.7,
            alert_cooldown_secs: 300, // 5 minutes
            enable_ethos: true,
        }
    }
}

/// Streaming inference engine for real-time monitoring
pub struct StreamingInference {
    config: StreamingConfig,
    ethos_guard: EthosGuard,
    patient_states: HashMap<String, PatientState>,
    feature_weights: Vec<(String, f64)>,
}

/// Internal state for a patient
#[derive(Debug, Clone)]
struct PatientState {
    last_update: i64,
    last_alert: Option<i64>,
    vital_history: Vec<VitalUpdate>,
    current_risk: f64,
}

impl StreamingInference {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            ethos_guard: EthosGuard::clinical_default(),
            patient_states: HashMap::new(),
            feature_weights: Vec::new(),
        }
    }

    /// Set feature weights from mRMR analysis
    pub fn set_feature_weights(&mut self, weights: Vec<(String, f64)>) {
        self.feature_weights = weights;
    }

    /// Process a single vital update
    pub fn process_update(&mut self, update: VitalUpdate) -> Result<(Option<InferenceResult>, Vec<Alert>)> {
        let patient_id = update.patient_id.clone();
        let timestamp = update.timestamp;
        
        // Get or create patient state
        let state = self.patient_states.entry(patient_id.clone()).or_insert(PatientState {
            last_update: 0,
            last_alert: None,
            vital_history: Vec::new(),
            current_risk: 0.0,
        });
        
        state.last_update = timestamp;
        state.vital_history.push(update.clone());
        
        // Keep only last 24 hours of history (assuming hourly updates)
        if state.vital_history.len() > 24 {
            state.vital_history.remove(0);
        }

        let patient_data = update.to_patient_data();
        let mut alerts = Vec::new();

        // Check Ethos guardrails
        if self.config.enable_ethos {
            match self.ethos_guard.check(&patient_data, ()) {
                EthosResult::Blocked(explanation) => {
                    alerts.push(Alert {
                        patient_id: patient_id.clone(),
                        alert_type: AlertType::EthosBlocked,
                        message: format!("Prediction blocked: {}", explanation.rule_violated),
                        severity: AlertSeverity::Warning,
                        timestamp,
                        triggering_values: HashMap::new(),
                    });
                    warn!("Patient {}: Prediction blocked by Ethos - {}", patient_id, explanation.rule_violated);
                    return Ok((None, alerts));
                }
                EthosResult::Allowed(_) => {}
            }
        }

        // Calculate sepsis risk based on feature weights
        let (risk_score, contributions) = self.calculate_risk(&patient_data);
        state.current_risk = risk_score;

        let inference_result = InferenceResult {
            patient_id: patient_id.clone(),
            timestamp,
            sepsis_risk: risk_score,
            risk_level: RiskLevel::from_score(risk_score),
            top_contributing_factors: contributions.clone(),
            confidence: self.calculate_confidence(&patient_data),
        };

        // Generate alerts if threshold exceeded
        if risk_score >= self.config.sepsis_alert_threshold {
            let should_alert = match state.last_alert {
                Some(last) => (timestamp - last) >= self.config.alert_cooldown_secs as i64,
                None => true,
            };

            if should_alert {
                state.last_alert = Some(timestamp);
                alerts.push(Alert {
                    patient_id: patient_id.clone(),
                    alert_type: AlertType::SepsisRisk,
                    message: format!("HIGH SEPSIS RISK: {:.1}%", risk_score * 100.0),
                    severity: if risk_score >= 0.9 { AlertSeverity::Emergency } else { AlertSeverity::Critical },
                    timestamp,
                    triggering_values: contributions.iter()
                        .take(3)
                        .map(|(k, v)| (k.clone(), *v))
                        .collect(),
                });
                info!("Patient {}: ALERT - Sepsis risk {:.1}%", patient_id, risk_score * 100.0);
            }
        }

        Ok((Some(inference_result), alerts))
    }

    /// Calculate risk score based on weighted features
    fn calculate_risk(&self, data: &PatientData) -> (f64, Vec<(String, f64)>) {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        let mut contributions = Vec::new();

        for (feature, weight) in &self.feature_weights {
            // Normalize weight
            total_weight += weight;

            // Get value from vitals or labs
            let value = data.get_vital(feature)
                .or_else(|| data.get_lab(feature));

            if let Some(v) = value {
                // Simple normalization (in real implementation, use proper scaling)
                let normalized = (v / 100.0).clamp(0.0, 1.0);
                let contribution = normalized * weight;
                weighted_sum += contribution;
                contributions.push((feature.clone(), contribution));
            }
        }

        let risk = if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        } else {
            0.5 // Default when no features available
        };

        contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        (risk, contributions)
    }

    /// Calculate confidence based on data availability
    fn calculate_confidence(&self, data: &PatientData) -> f64 {
        let mut available = 0;
        let mut total = 0;

        for (feature, _) in &self.feature_weights {
            total += 1;
            if data.get_vital(feature).is_some() || data.get_lab(feature).is_some() {
                available += 1;
            }
        }

        if total > 0 {
            available as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Async streaming processor using channels
pub struct AsyncStreamProcessor {
    input_tx: mpsc::Sender<VitalUpdate>,
    output_rx: mpsc::Receiver<(Option<InferenceResult>, Vec<Alert>)>,
}

impl AsyncStreamProcessor {
    /// Create a new async processor with background task
    pub fn spawn(config: StreamingConfig, feature_weights: Vec<(String, f64)>) -> Self {
        let (input_tx, mut input_rx) = mpsc::channel::<VitalUpdate>(100);
        let (output_tx, output_rx) = mpsc::channel(100);

        tokio::spawn(async move {
            let mut engine = StreamingInference::new(config);
            engine.set_feature_weights(feature_weights);

            while let Some(update) = input_rx.recv().await {
                match engine.process_update(update) {
                    Ok(result) => {
                        if output_tx.send(result).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Inference error: {}", e);
                    }
                }
            }
        });

        Self { input_tx, output_rx }
    }

    /// Send an update for processing
    pub async fn send(&self, update: VitalUpdate) -> Result<()> {
        self.input_tx.send(update).await?;
        Ok(())
    }

    /// Receive the next result
    pub async fn recv(&mut self) -> Option<(Option<InferenceResult>, Vec<Alert>)> {
        self.output_rx.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_from_score() {
        assert_eq!(RiskLevel::from_score(0.1), RiskLevel::Low);
        assert_eq!(RiskLevel::from_score(0.4), RiskLevel::Moderate);
        assert_eq!(RiskLevel::from_score(0.6), RiskLevel::High);
        assert_eq!(RiskLevel::from_score(0.9), RiskLevel::Critical);
    }

    #[test]
    fn test_streaming_inference_basic() {
        let config = StreamingConfig::default();
        let mut engine = StreamingInference::new(config);
        engine.set_feature_weights(vec![
            ("HR".to_string(), 1.0),
            ("MAP".to_string(), 0.8),
        ]);

        let update = VitalUpdate {
            patient_id: "P001".to_string(),
            timestamp: 1000,
            vitals: [
                ("HR".to_string(), Some(85.0)),
                ("MAP".to_string(), Some(70.0)),
            ].into_iter().collect(),
            labs: HashMap::new(),
        };

        let (result, alerts) = engine.process_update(update).unwrap();
        assert!(result.is_some());
    }
}
