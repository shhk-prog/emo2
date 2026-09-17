# 論文統合の完了報告 (Walkthrough)

ご指摘を受け、`/mnt/nas/home/hiromi/src/emo/v1` と `/mnt/nas/home/hiromi/src/emo/v2` の草稿に含まれていた詳細な実験設定、数式、具体的な検証結果（全表および数値）をすべて網羅するように `/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` を大幅に加筆・拡張しました。

## 実施した主な加筆内容

### 1. 実験設定の詳細化
- AIPsy-Affectデータセットの抽出基準（Strict Matched Subset）や具体的なデータ例（Grief: Affective vs Neutral）を復元しました。
- 尤度プロトコル（Sequence Likelihood Protocol）の具体的な対数尤度の算出式とソフトマックス正規化の式を明記しました。

### 2. 定量的な結果・表の完全復元
- **予備実験（GPT-4o）**: EmoBankに対する完全な中立化（98.6%）を示す行動的反応性の結果表（表1）を復元しました。
- **Cross-decoding (H2)**: Ridge Alignmentによる部分的な回復（$R^2 \approx 0.58$）を示す「Cross-Decoding Performance」の表（表2）と、RSAによる負の相関値を追加しました。
- **因果介入 (Patching & Ablation)**: 83.3%のRecoveryスコアや層4での42.26%消去といった詳細な数値を記載しました。
- **コンポーネント介入 (H4)**: $|\Delta WD_V|$ によるTop Componentsのランキング表（表3）、および同一プロンプト条件下での後期残差置換テストの結果表（表4）を復元しました。
- **8-Condition Swap**: 最終出力段における8条件のResidual / RMSNorm / Unembeddingスワップ解析の詳細な結果表（表5）を復元しました。

### 3. Appendixの拡充
ご指示通り、以下の項目をすべてAppendixとして詳細に記載しています。
- v1のコード・ディレクトリ構造とその役割
- Batch size最適化の記述
- Scaling ratioの発散結果（表を復元）
- 開発Phase番号に関する経緯の説明
- ベースライン、感情認識、事後自己報告、Affective Receptionの全プロンプトの完全なテキスト

## 確認事項
本作業により、以下のファイルが更新・作成されています。
- [統合論文の完成版 (v3/docs/paper.md)](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md) (情報量を大幅に加筆)
- [実装計画書 (docs/integrate_v1_v2_paper/implementation_plan.md)](file:///mnt/nas/home/hiromi/src/emo/docs/integrate_v1_v2_paper/implementation_plan.md)
- [タスクリスト (docs/integrate_v1_v2_paper/task.md)](file:///mnt/nas/home/hiromi/src/emo/docs/integrate_v1_v2_paper/task.md)
- ウォークスルー (本ファイル)

元のv1・v2草稿と同等の解像度を持った詳細な論文となっておりますので、内容をご確認ください。
