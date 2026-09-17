# V1 Phase A Probing Summary Report: Qwen/Qwen2.5-1.5B

- **Model**: `Qwen/Qwen2.5-1.5B`
- **Layers Evaluated**: 29
- **Dataset Evaluated**: `both`

## EmoBank (E1: Shared Decodability & E2: Shared Geometry)

- **Reader V Peak Layer**: Layer 25 ($R^2 = 0.371$)
- **Self V Peak Layer**: Layer 23 ($R^2 = 0.375$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 2$ layers

### Geometry Patterns Across Layers:
- **Alignable Geometry**: 24 / 29 layers (82.8%)
- **Shared Geometry**: 4 / 29 layers (13.8%)
- **Directly non-transferable / poorly alignable representation**: 1 / 29 layers (3.4%)

## AIPsy-Affect (E1: Shared Decodability & E2: Shared Geometry)

- **Reader Condition Peak AUC**: Layer 17 ($	ext{AUC} = 0.977$)
- **Self Condition Peak AUC**: Layer 17 ($	ext{AUC} = 0.975$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 0$ layers

