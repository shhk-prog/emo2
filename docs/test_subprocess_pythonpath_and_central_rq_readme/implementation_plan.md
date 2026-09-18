# 実装計画: テスト環境非依存化 (PYTHONPATH) と root README への Central RQ / 4-Stage 統合

## 概要
Behavioral → V1 → V2 → V3 の本番全再実行に向け、残る2点（① entrypoint 引数互換性テストにおける subprocess の `PYTHONPATH` 補完による clone 直後テスト通過の保証、② root `README.md` 冒頭への Central RQ・4-Stage 概念対応表・論文構成・Contribution・Core Thesis の統合記載）を実施します。

---

## 提案する変更内容

### 1. テスト環境・サブプロセス呼び出しの堅牢化

#### [MODIFY] [`tests/test_production_entrypoints.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_production_entrypoints.py)
- `test_all_dispatched_commands_argparse_compatibility` 内で各スクリプトに対して `subprocess.run([sys.executable, script_path, "--help"], ...)` を呼び出す際、現在実行中のプロセスの環境変数だけでなく、明示的に `src/` を `PYTHONPATH` に付加した `env` およびリポジトリルートを `cwd` として渡します。
- これにより、`pip install -e .` がされていない環境や clone 直後の環境で `pytest` を実行した場合でも、`v1/primary/run_phase_a.py` などのサブプロセスが確実に `affective_empathy_eval` パッケージを解決できるようにします。

#### [MODIFY] [`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- `run_command(cmd)` においても同様に、呼び出し先サブプロセスへ `PYTHONPATH`（`repo_root / "src"` を含む）を渡すよう補強し、統合ランナー経由でどのスクリプトが呼ばれた場合でも環境差異によるインポートエラーを未然に防止します。

---

### 2. root README 冒頭のドキュメント拡充

#### [MODIFY] [`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md)
ユーザーから提示された全体構想・論理構造に基づき、README の冒頭を以下の通り体系化します：

1. **Central Research Question (中心 RQ)**:
   > **Central RQ**: *How are affect-relevant internal representations coupled to LLM self-reports, and how does this relationship vary across tasks, post-training, and stages of computation?*  
   > （LLMの情動関連内部表現は自己報告とどのように結びついており、その関係はタスク、post-training、計算過程を通じてどのように変化するのか。）

2. **4-Stage 概念対応表（一続きの証拠階層）**:
   ```text
   Covariation  →  Representation / Causality  →  Reorganization  →  Utilization
   (Behavioral)               (V1)                         (V2)                (V3)
   ```
   | Stage | 科学的問い (Paper Section) | 概念的役割 | 比較・検証軸 |
   |---|---|---|---|
   | **Behavioral** | 3. Behavioral Characterization | **Covariation** (相関・導入現象) | 出力分布。認識と自己報告は独立セッション |
   | **V1** | 4. Internal Representation and Causal Sharing | **Representation / Causality** (表現・因果共有) | 同一モデル内の Reader ↔ Self |
   | **V2** | 5. Post-training Reorganization | **Reorganization** (事後学習再編) | 同一ファミリーの Base ↔ Instruct |
   | **V3** | 6. From Representation to Causal Utilization | **Utilization** (因果的利用可能性) | Instruct 側の層 × 生成段階 |

3. **中心的主張（Core Thesis）**:
   > 「LLMの自己報告は、単なる出力上の模倣でも、内部でデコード可能な情動情報の直接的な読み出しでもない。自己報告は情動関連内部表現と系統的に結びつくが、その結びつきは部分的かつタスク依存であり、post-trainingによって再編され、計算過程の特定の位置で初めて因果的利用可能性を持つ。」

4. **3本の学術的貢献（Three Main Contributions）**:
   - **Contribution 1 (Behavioral + V1)**: Reader PredictionとSelf-Reportがcovaryし、その背後に部分的に共有されたaffect-relevant representationと因果機構が存在することを示す。
   - **Contribution 2 (V2)**: Post-trainingがその情報を単純に消去するのではなく、表現幾何・Reader/Self共有性・因果利用を再編することを示す。
   - **Contribution 3 (V3)**: Decodable informationとcausally utilized informationを分離し、自己報告への因果力がどの層・生成段階で現れるかを検証する。

5. **729 VAD vs 81 VA の位置づけの明文化**:
   - Behavioral / V1: original VAD protocol を保持（過去知見・アノテーションプロトコルとの整合）
   - V2 / V3: causal sweep の計算量を抑え、Primary endpoint を VA へ限定
   - Dominance 次元は Behavioral / V1 の補助次元とし、最終主張には直接関与させず Appendix で補足する位置づけを明記。

---

## 検証計画

### 自動テスト
1. **`tests/test_production_entrypoints.py` の単独テスト**:
   - `.venv/bin/pytest -q tests/test_production_entrypoints.py` を実行。
2. **全テストスイートの実行**:
   - `.venv/bin/pytest -q` を実行し、全テスト（61件）が pass することを確認。
3. **`PYTHONPATH` 補完の検証**:
   - サブプロセスのテストで実際に環境変数 `PYTHONPATH` が効いて `v1/primary/run_phase_a.py --help` が正常に終了することを確認。

### 手動確認
- `README.md` の markdown 表示崩れやリンク切れがないことを目視確認。
- ユーザー指示ルールに従い、`docs/test_subprocess_pythonpath_and_central_rq_readme/` 配下に `task.md`, `implementation_plan.md`, `walkthrough.md` を保存。
