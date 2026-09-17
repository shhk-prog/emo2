# V1 Phase A Probing Summary Report: google/gemma-2-2b

- **Model**: `google/gemma-2-2b`
- **Layers Evaluated**: 27
- **Dataset Evaluated**: `both`

## EmoBank (E1: Shared Decodability & E2: Shared Geometry)

- **Reader V Peak Layer**: Layer 26 ($R^2 = -0.006$)
- **Self V Peak Layer**: Layer 26 ($R^2 = 0.007$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 0$ layers

### Geometry Patterns Across Layers:
- **Alignable Geometry**: 26 / 27 layers (96.3%)
- **Directly non-transferable / poorly alignable representation**: 1 / 27 layers (3.7%)

## AIPsy-Affect (E1: Shared Decodability & E2: Shared Geometry)

- **Reader Condition Peak AUC**: Layer 14 ($	ext{AUC} = 0.863$)
- **Self Condition Peak AUC**: Layer 14 ($	ext{AUC} = 0.875$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 0$ layers

