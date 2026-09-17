# V1 Phase A Probing Summary Report: mistralai/Mistral-7B-Instruct-v0.1

- **Model**: `mistralai/Mistral-7B-Instruct-v0.1`
- **Layers Evaluated**: 33
- **Dataset Evaluated**: `both`

## EmoBank (E1: Shared Decodability & E2: Shared Geometry)

- **Reader V Peak Layer**: Layer 21 ($R^2 = 0.487$)
- **Self V Peak Layer**: Layer 18 ($R^2 = 0.480$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 3$ layers

### Geometry Patterns Across Layers:
- **Shared Geometry**: 24 / 33 layers (72.7%)
- **Alignable Geometry**: 8 / 33 layers (24.2%)
- **Directly non-transferable / poorly alignable representation**: 1 / 33 layers (3.0%)

## AIPsy-Affect (E1: Shared Decodability & E2: Shared Geometry)

- **Reader Condition Peak AUC**: Layer 16 ($	ext{AUC} = 0.991$)
- **Self Condition Peak AUC**: Layer 16 ($	ext{AUC} = 0.992$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 0$ layers

