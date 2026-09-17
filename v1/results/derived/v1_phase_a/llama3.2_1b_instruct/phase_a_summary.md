# V1 Phase A Probing Summary Report: meta-llama/Llama-3.2-1B-Instruct

- **Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Layers Evaluated**: 17
- **Dataset Evaluated**: `both`

## EmoBank (E1: Shared Decodability & E2: Shared Geometry)

- **Reader V Peak Layer**: Layer 10 ($R^2 = 0.326$)
- **Self V Peak Layer**: Layer 10 ($R^2 = 0.390$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 0$ layers

### Geometry Patterns Across Layers:
- **Alignable Geometry**: 16 / 17 layers (94.1%)
- **Directly non-transferable / poorly alignable representation**: 1 / 17 layers (5.9%)

## AIPsy-Affect (E1: Shared Decodability & E2: Shared Geometry)

- **Reader Condition Peak AUC**: Layer 10 ($	ext{AUC} = 0.982$)
- **Self Condition Peak AUC**: Layer 9 ($	ext{AUC} = 0.979$)
- **Peak Layer Displacement**: $|l^*_R - l^*_S| = 1$ layers

