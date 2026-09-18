# V1 Phase A Probing Summary Report: Qwen/Qwen2.5-1.5B-Instruct

- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Layers Evaluated**: 29
- **Dataset Evaluated**: `both`

## EmoBank (E1: Shared Decodability & E2: Shared Geometry)

- **Reader V Peak Layer**: Layer 14 ($R^2 = 0.234$)
- **Self V Peak Layer**: Layer 14 ($R^2 = 0.288$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 0$ layers

### Geometry Patterns Across Layers:
- **Alignable Geometry**: 27 / 29 layers (93.1%)
- **Directly non-transferable / poorly alignable representation**: 1 / 29 layers (3.4%)
- **Shared Geometry**: 1 / 29 layers (3.4%)

## AIPsy-Affect (E1: Shared Decodability & E2: Shared Geometry)

- **Reader Condition Peak AUC**: Layer 18 ($	ext{AUC} = 0.964$)
- **Self Condition Peak AUC**: Layer 14 ($	ext{AUC} = 0.963$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 4$ layers

