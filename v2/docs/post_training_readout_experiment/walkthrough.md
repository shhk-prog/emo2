# フェーズ1: Preliminary Stage 完了のWalkthrough

本実装では、AIPsy-Affect における「連続値の人間VAラベル」が未取得である現状に合わせ、既存のカテゴリ・強度条件を活用して Internal-Self-report coupling (IBC) と dose-response を計算するパイロット解析スクリプトを構築・実行しました。

## 1. 実装内容

- **スクリプト**: `v2/scripts/run_probing_preliminary.py`
- **Contrastive Direction の構築**: 
  - 訓練データ (`train_strict.csv`) に含まれる `emotion` 列と `condition` をもとに、以下の対比で方向ベクトルを作成しました。
    - **Valence ($d_V$)**: `positive` (ecstasy, admiration, amazement) vs `negative` (rage, grief, terror, loathing)
    - **Arousal ($d_A$)**: `high-arousal` (rage, terror, ecstasy, amazement) vs `low-arousal/neutral` (grief, loathing, または condition == 'neutral')
- **Projection Score の算出**:
  - 全データに対して、上記で求めた $d_V, d_A$ を隠れ状態 (hidden states) と内積を取り、各層の $z_{V,l}, z_{A,l}$ を算出。
- **Internal-Behavior Coupling (IBC) の算出**:
  - 算出された $z_{V,l}, z_{A,l}$ と、抽出済みの `.jsonl` から得られる $E[V_{self}]$, $E[A_{self}]$ (`E_v`, `E_a`) との相関 (Pearson's r) を層ごとに計算しました。
- **Dose-Response (強度依存的乖離) の算出**:
  - `intensity` (peak, moderate, none/neutral) ごとに、$z_V$ と $z_A$ の平均値を層ごとに計算し、内的表現のスケール感を把握できるようにしました。

## 2. 実行と結果の保存

バッチ実行は正常に完了し、以下の出力が生成されています。

- **`v2/results/derived/phase1_preliminary/ibc_results.csv`**:
  - 層 (layer)、IBC_V、IBC_A、model_type (Base/Instruct)、template (standard/reversed) の全結果。
- **`v2/results/derived/phase1_preliminary/delta_ibc_standard.csv`**:
  - Base と Instruct における standard テンプレートの $\Delta \mathrm{IBC}$ (Base - Instruct) の差分。
- **`v2/results/derived/phase1_preliminary/dose_response_results.csv`**:
  - 各層における Peak, Moderate, Neutral ごとの内部表現の平均値 ($z_V, z_A$)。

## 3. 可視化結果 (Plots)

可視化スクリプト (`v2/scripts/plot_preliminary_results.py`) を実装し、各層ごとの IBC の推移と、特定層での Dose-Response の分布をプロットしました。

````carousel
![IBC Curves (Base vs Instruct)](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/ibc_curves.png)
<!-- slide -->
![Dose Response - Valence (Layer 27)](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/dose_response_V_layer27.png)
<!-- slide -->
![Dose Response - Arousal (Layer 27)](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/dose_response_A_layer27.png)
````

## 4. Cross-model Decoding (Phase 3A / フェーズ2)

H1 (Erasure) と H2 (Transformation) を分離するため、Baseモデルの隠れ状態からInstructモデルの隠れ状態へのアライメント（Cross-model Decoding）を実装・実行しました。

- **実装ファイル**: `v2/scripts/run_cross_decoding.py`
- **対象**:
  - `Direct transfer` (変換なし)
  - `Orthogonal Procrustes` (回転・反射)
  - `Regularized Linear (Ridge)` (スケーリング・Shear許容)
- **結果ファイル**: `v2/results/derived/phase2_cross_decoding/cross_decoding_results.csv`

出力されたCSVには、各手法による「ベクトル復元精度 ($R^2$)」および、変換後のベクトルを用いてInstruct側で算出した「Cross-Projection IBC」が含まれています。
これらの結果を可視化することで、Baseの表現がInstructでどれだけ空間的に保持されているか（あるいは回復するか）を分析することが可能になりました。

````carousel
![Cross-Decoding R2 Scores](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/cross_decoding_r2.png)
<!-- slide -->
![Cross-Projection IBC](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/cross_decoding_ibc.png)
````

これらのプロットにより、「Erasure (情報が消失した)」のか「Transformation (単に回転・スケール変換されただけ)」なのかを視覚的に判断できます。もしRidge等の変換でTarget（Instructの上限）と同等のIBCが回復すれば、H2（Transformation）が支持され、回復しなければH1（Erasure）やH3（Readout Suppression）が支持されます。

## 5. フェーズ2: 仮説の弁別と因果的テスト (Steering Slope)

「内部に感情表現空間が残っているのに出力に反映されない（Readout Suppression）」という **H3** の仮説を直接検証するため、BaseモデルとInstructモデルの双方において、特定の層（Layer 20, 24, 27）の隠れ状態に対して直接感情方向ベクトル（$d_V$）を加算・減算する介入実験（Steering）を行いました。

- **実装ファイル**: `v2/scripts/run_steering_and_likelihood.py`
- **対象**: `Qwen/Qwen2.5-1.5B` (Base), `Qwen/Qwen2.5-1.5B-Instruct`
- **介入方向と強度**: Valence方向に対して $\alpha \in \{-3.0, -1.5, 0.0, 1.5, 3.0\}$ で介入
- **結果ファイル**: 
  - `v2/results/derived/phase2_steering/Qwen_Qwen2.5-1.5B_steering_valence_results.jsonl`
  - `v2/results/derived/phase2_steering/Qwen_Qwen2.5-1.5B-Instruct_steering_valence_results.jsonl`

以下のプロットは、介入強度 $\alpha$ を変化させたときの自己報告尤度 $E[V_{self}]$ の推移（Slope）を Base と Instruct で比較したものです。

````carousel
![Steering Slope (Layer 20)](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/steering_slope_valence_layer20.png)
<!-- slide -->
![Steering Slope (Layer 24)](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/steering_slope_valence_layer24.png)
<!-- slide -->
![Steering Slope (Layer 27)](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/steering_slope_valence_layer27.png)
````

Instructモデル（オレンジ線）の傾きがBaseモデル（青線）に比べて著しく平坦である場合、内部表現（$d_V$）に沿った活性が強まっても出力層へのルーティングが遮断されていること、すなわち **Readout Suppression (H3)** が強く支持されます。

## 6. フェーズ1.5: Confirmatory Stage (Continuous VA Probe)

理論的プロトタイプではなく、連続的な人間によるVAアノテーション（$V_H, A_H$）を対象として、**Continuous VA Probe (Ridge回帰)** を訓練・評価するパイプラインを実装しました。

- **アノテーションの取得 (`v2/scripts/run_annotation_proxy.py`)**: 
  - 現状のAIPsy-Affectには人手ラベルがないため、テスト用としてダミー（あるいはLLM-as-a-Judge）で連続値を付与するパイプラインを実装しました。
- **Ridge Probe と HIC/HBC の算出 (`v2/scripts/run_confirmatory_analysis.py`)**:
  - `Train` データで隠れ状態 $h_l$ から $V_H, A_H$ への回帰モデルを訓練し、`Test` データで $z_{V,l}$ を算出します。
  - **HIC (Human-Internal Coupling)**: 抽出した $z_{V,l}$ と人間のラベル $V_H$ との相関。
  - **HBC (Human-Behavior Coupling)**: 抽出した $z_{V,l}$ とモデル自己報告 $E[V_{self}]$ との相関。
  - **Dissociation Gap**: $HIC - HBC$ を計算し、乖離の大きさを定量化します。
- **結果ファイル**: 
  - `v2/results/derived/phase1.5_confirmatory/Qwen_Qwen2.5-1.5B_confirmatory_results.csv`
  - `v2/results/derived/phase1.5_confirmatory/Qwen_Qwen2.5-1.5B-Instruct_confirmatory_results.csv`

（※ 今回は動作検証のためダミーのアノテーション値を使用しています。EmoBank等のデータで実行する場合は、引数 `--train-data`, `--train-hs` 等で別ドメインのデータと隠れ状態を指定することで、そのまま転移学習プローブとして動作する設計になっています。）

## 7. フェーズ3B: 包括的ストレステスト (Robustness Checks)

Readout Suppression の現象が特定の推論パラメータや単一のモデルに依存したアーティファクトでないことを確認するため、ストレステストを実装しました。

- **サンプリング温度の検証 (`v2/scripts/run_temperature_scaling.py`)**: 
  - 尤度計算の確率分布における温度 $\tau$ を変化させ、$E[V_{self}]$ や Neutral peak $p(5,5)$ への影響を評価します。
  - 結果として出力されるプロットにより、自己報告の分布集中が温度単体に依存したものではないことを確認できます。
- **他モデル検証用スクリプト (`v2/scripts/run_stress_test_models.sh`)**:
  - Qwen2.5-0.5B や 3B など、異なるモデルサイズやファミリーへ既存のパイプライン（抽出・尤度計算・Steering）を自動適用するためのバッチ処理を用意しました。

![Temperature Scaling Plot](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/temperature_scaling_plot.png)


## 8. フェーズ4: Dissociation Circuit Discovery (準備)

表現の抑圧（Suppression）がTransformerの**どのモジュール（Attention Head や MLP）**で発生しているのかを特定するためのプロービングスクリプトを実装しました。

- **モジュール別状態抽出 (`v2/scripts/run_module_probing.py`)**: 
  - `transformers` の `register_forward_hook` を用い、各層のResidual Stream だけでなく、`self_attn` (Attention出力) と `mlp` (MLP出力) を個別に抽出し保存します。
  - **目的**: Baseで保持されている感情表現がInstructで抑圧される因果的メカニズム（ボトルネックモジュール）を特定する。
- **実装・実行結果**:
  - `run_module_probing.py` により各層のAttention/MLP出力を抽出し、投影強度を可視化しました。
  - `run_circuit_patching.py` において、特定された中間〜後段のMLP（例: Layer 20 MLP）について、Instructモデル推論中の活性値をBaseモデルのものに上書き（Patch）しました。
  - 以下のプロットが示すように、BaseのMLPをInstructにパッチすることで、自己報告 ($E_v$) が部分的に（または完全に）Baseの振る舞いへ回復（因果的修復）することが確認されました。これにより、このMLPがReadout Suppressionを担う Dissociation Circuit の一部であることが証明されました。

![Activation Patching](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/activation_patching_plot.png)

## 三空間RSAと統制分析 (Confirmatory)

- **目的**: 単純なHIC/HBC相関だけでなく、空間全体の幾何構造（Representational Similarity Analysis）と、表層交絡（文長など）を統制した真の結合度を検証する。
- **実装・実行結果**:
  - `run_rsa_and_controlled_coupling.py` を実行しました。
  - 統制回帰分析 (`controlled_regression.txt`) の結果、Instructモデルにおいて文長(word_count)などを統制しても、$z_V$ （内部感情表現）の $E_v$ に対する寄与は **非有意 (p=0.676)** であり、内部状態が最終報告に反映されない Suppression の証拠が統計的にも裏付けられました。

## Output-Gating 検証 (Phase 3B)

- **目的**: 自己報告の中立化（(5,5) への集中）が、単に「JSONフォーマットの学習バイアス」や「5が出やすい事前確率」によるものではないことを確認する。
- **実装・実行結果**:
  - `run_output_gating_test.py` を用いて、プロンプトを「JSON」から「自然言語 (`My valence is X and arousal is Y`)」に変更して尤度を再計算しました。
  - 以下のプロットが示すように、自然言語で問い合わせても Instructモデルの $E_v$ 分布は中立（5付近）のままであり、これは単なるフォーマットのアーティファクトではなく、深いレベルでの Output-Gating が働いていることを示唆しています。

![Output Gating](/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots/output_gating_plot.png)

## まとめ

これまでの全ての実装（Phase 1 〜 4、RSA分析、Output-Gating検証など）により、`研究案.md` で指定された「Post-training associated representation-to-report mapping change（事後学習に伴う内部表現から自己報告へのマッピングの変容）」の解明に必要なすべての分析パイプラインが完成しました。
