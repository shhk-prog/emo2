# 実装計画: H1/H2仮説の再定式化とCross-Decodingの論理的再構成

## 1. 概要
本計画は、ユーザーから提案された「H1 Complete Erasure から H1 Representational Replacement / Loss of Base Geometry への昇華」を体系化し、論文・ドラフトおよび関連文書に展開するための論理設計である。

## 2. 核心的な論理構造の比較

### 従来の定義（問題点：v1との重複・藁人形化）
- **H1 (Complete Erasure)**: 事後学習により、Instructモデルの内部空間から感情情報自体が完全に消去された。
  - *問題点*: v1ですでにWithin-model Probeにより $I(\text{Instruct};\text{affect}) > 0$（ROC-AUC > 97%）が示されており、v1の結果を前提にすると自明に棄却される藁人形仮説になってしまう。
- **H2 (Transformation)**: 感情情報は維持されるが、表現空間が非直交的に歪められた。

### 再定式化後の定義（美しい対比・Cross-Decodingの必然性）
- **H1: Representational Replacement / Loss of Base Geometry (Base表現の喪失・置換)**:
  - 事後学習（Post-training）によって、Baseモデルで形成されていた感情表現幾何 $h^{\text{Base}}$ が失われ、新しい別の表現空間 $h^{\text{Instruct}}$ へ置換された。Base表現との写像的対応関係（Alignment）は保持されない。
- **H2: Representational Transformation (表現幾何の変換・連続的保持)**:
  - Baseモデルの感情表現は事後学習後も情報として保持されており、表現空間が非直交的に歪められているものの、適切な写像（$h^{\text{Base}} \approx W h^{\text{Instruct}}$）によって予測性能が回復する形で対応関係が維持されている。

## 3. v1からv2への接続ストーリー（キラーセンテンス）
> 「V1ではInstructモデル内部にも感情関連情報が存在することを確認した。そこでV2では、Post-trainingによってBaseモデルの感情表現が失われ別の表現へ置換されたのか（H1）、それともBaseの情報を保持したまま表現空間が変換されたのか（H2）をCross-Decodingによって検証する。」

## 4. 適用範囲と今後の反映方針
1. `docs/v2_hypothesis_redefinition_cross_decoding/` に詳細な解説ドキュメントを保存。
2. 今後、`v2/README.md`, `v2/docs/post_training_readout_experiment/paper_draft.md` や `v3` の関連セクションを更新する際の基準定義とする。
