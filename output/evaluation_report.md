# Evaluation Report

## AI-Driven Hybrid FCM Brain Tumor MRI Segmentation

## Ablation Study

| Model | DSC | IoU | HD95 | Sens | Spec |
|-------|-----|-----|------|------|------|
| E: Full Pipeline | 0.6448±0.0020 | 0.5208 | 20.60 | 0.9223 | 0.9750 |
| D: AI+PCA+FCM | 0.5278±0.0016 | 0.3966 | 22.95 | 0.8619 | 0.9551 |
| C: CNN+FCM | 0.4540±0.0014 | 0.3243 | 23.94 | 0.8072 | 0.9373 |
| B: PCA+FCM | 0.4099±0.0007 | 0.2836 | 24.36 | 0.7699 | 0.9246 |
| A: FCM Standard | 0.3372±0.0005 | 0.2208 | 25.05 | 0.6972 | 0.8993 |

## Robustness Test

| Noise Type | Clean DSC | Noisy DSC | Degradation |
|------------|-----------|-----------|-------------|
| gaussian_0.05 | 0.6464 | 0.5039 | 22.05% |
| gaussian_0.10 | 0.6464 | 0.4092 | 36.70% |
| salt_pepper_0.03 | 0.6464 | 0.5521 | 14.58% |
| salt_pepper_0.07 | 0.6464 | 0.4638 | 28.25% |

## Statistical Tests

- **D_vs_E**: wilcoxon_signed_rank, p=0.000000, ✓ Significant
- **B_vs_D**: wilcoxon_signed_rank, p=0.000000, ✓ Significant
