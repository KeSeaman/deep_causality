# Example diagnostic reports (hospital B, causal-driver detector)

## True alert (septic stay, labelled hour)

### Patient 117886, hour 28 (septic stay: yes; label this hour: 1)

- Risk score: 0.064 (alert threshold 0.060) -> ALERT
- Guardrails: allowed
- Largest contributions (log-odds, SHAP):
  - Temp = 38.50: +1.386
  - Hgb = 8.00: +0.367
  - HCO3 = missing: -0.219
  - Resp = 23.00: +0.172
  - BUN = 15.00: -0.164

## False alarm (never-septic stay)

### Patient 112741, hour 1 (septic stay: no; label this hour: 0)

- Risk score: 0.081 (alert threshold 0.060) -> ALERT
- Guardrails: allowed
- Largest contributions (log-odds, SHAP):
  - Resp = 26.00: +0.665
  - Fibrinogen = 516.00: +0.558
  - Temp = missing: -0.302
  - HR = 109.00: +0.210
  - Glucose = 226.00: +0.201

## Blocked by guardrails

### Patient 110241, hour 11 (septic stay: no; label this hour: 0)

- Risk score: 0.068 (alert threshold 0.060) -> ALERT
- Guardrails: BLOCKED (ETHOS-002: more than 50% of vitals and labs missing)
- Largest contributions (log-odds, SHAP):
  - Temp = 38.40: +0.948
  - Hgb = 7.80: +0.367
  - HCO3 = missing: -0.250
  - HR = 99.00: +0.170
  - BUN = missing: +0.166
