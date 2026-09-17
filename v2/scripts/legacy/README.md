# V2 Legacy Scripts Archive

このディレクトリ (`v2/scripts/legacy/`) は、V2 開発初期に用いられた探索的スクリーニング、旧 SAE パッチング、旧内省的アクセス可能性検証、疑似ラベル生成、プレースホルダー結合解析などのスクリプトを隔離・保存したものです。

## 現行の V2 主要パイプラインとの対応

論文の Primary 分析（RQ1〜RQ4）および厳密な再現性実験には、`v2/scripts/` 直下の新スクリプトを使用してください。

| リサーチクエスチョン | 現行 Primary スクリプト (`v2/scripts/`) | 旧・探索的スクリプト (`v2/scripts/legacy/`) | 備考 |
|---|---|---|---|
| **RQ1 (Representation)** | `run_v2_2x2_cross_decoding.py` | `run_strict_cross_decoding.py`, `run_cross_decoding.py` | 2×2 (Task: Recognition/Self-Report × Prompt: Direct/CoT) 線形・非線形クロスデコーディング |
| **RQ2 (Causal Map)** | `run_v2_2x2_causal_map.py` | `run_patching_screening.py`, `run_circuit_patching.py`, `run_strict_causal_scrubbing.py` | 2×2 条件での causal patching マッピング |
| **RQ3 (Controlled Coupling)** | `run_mixed_effects_coupling.py` | `run_rsa_and_controlled_coupling.py` (placeholder 方向含むため隔離) | 刺激共変量統制下の表現・因果結合（混合効果モデルによる傾き評価） |
| **RQ4 (Recovery Patching)** | `run_v2_recovery_patching.py` | `run_output_gating_test.py`, `run_sae_patching.py`, `run_introspective_accessibility_test.py`, `run_unembedding_norm_swap.py` | Base $\to$ Instruct のフォーマット統制（Instruct matched-plain control）リカバリーパッチング |
| **除外・探索的** | - | `run_annotation_proxy.py` | 乱数疑似ラベル生成スクリプト（論文解析から明確に除外） |

旧スクリプトは履歴追跡および再現性の検証目的でのみ参照し、主実験結果の生成には使用しないでください。
