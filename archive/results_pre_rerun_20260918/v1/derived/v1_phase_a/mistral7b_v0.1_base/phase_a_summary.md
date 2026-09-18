# V1 Phase A Probing Summary Report: mistralai/Mistral-7B-v0.1

- **Model**: `mistralai/Mistral-7B-v0.1`
- **Layers Evaluated**: 33
- **Dataset Evaluated**: `both`

## EmoBank (E1: Shared Decodability & E2: Shared Geometry)

- **Reader V Peak Layer**: Layer 32 ($R^2 = 0.172$)
- **Self V Peak Layer**: Layer 32 ($R^2 = 0.197$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 0$ layers

### Geometry Patterns Across Layers:
- **Alignable Geometry**: 23 / 33 layers (69.7%)
- **Shared Geometry**: 9 / 33 layers (27.3%)
- **Directly non-transferable / poorly alignable representation**: 1 / 33 layers (3.0%)

## AIPsy-Affect (E1: Shared Decodability & E2: Shared Geometry)

- **Reader Condition Peak AUC**: Layer 11 ($	ext{AUC} = 0.977$)
- **Self Condition Peak AUC**: Layer 15 ($	ext{AUC} = 0.980$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 4$ layers

