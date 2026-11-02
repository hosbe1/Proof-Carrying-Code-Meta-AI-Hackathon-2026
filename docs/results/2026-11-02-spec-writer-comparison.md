# Run comparison

| run | model | mode | dataset | n | rate | cost (USD) |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-02-48962e5-spec-spark | muse-spark-1.3 | contributor | humanevalfix/test | 39 | correct_rate 0.090 (95% CI 0.044-0.174) | 0.10 |
| 2026-11-02-48962e5-spec-wonder | muse-wonder-1.0 | standard | humanevalfix/test | 39 | correct_rate 0.077 (95% CI 0.036-0.158) | 8.32 |
| 2026-11-02-48962e5-spec-glimmer | glimmer | standard | humanevalfix/test | 39 | correct_rate 0.051 (95% CI 0.020-0.125) | 0.55 |

## Paired differences

### 2026-11-02-48962e5-spec-spark vs 2026-11-02-48962e5-spec-wonder

- Only in 2026-11-02-48962e5-spec-spark (1): humanevalfix/Python/15

### 2026-11-02-48962e5-spec-spark vs 2026-11-02-48962e5-spec-glimmer

- Only in 2026-11-02-48962e5-spec-spark (1): humanevalfix/Python/15

### 2026-11-02-48962e5-spec-wonder vs 2026-11-02-48962e5-spec-glimmer

- Both solve the same tasks (2 of 39).
