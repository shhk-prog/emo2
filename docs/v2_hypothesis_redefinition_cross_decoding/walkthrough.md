# 修正内容・解説の確認 (Walkthrough): H1/H2仮説の再定式化とCross-Decodingの理論的意義

## 1. 実施概要
ユーザーからの極めて洞察に富んだ提案に基づき、v1とv2の接続ロジックを整理し、以下の点を明確化しました。
1. **なぜ従来の「H1 Complete Erasure」は不十分だったのか**
2. **「H1: Representational Replacement」対「H2: Representational Transformation」という新定義の優位性**
3. **Cross-Decodingを行う論理的必然性と実験的証明の構造**

---

## 2. 比較対照表

| 項目 | 従来の定式化 | 新しい再定式化（提案） | 改善の理由・効果 |
|---|---|---|---|
| **H1の名称** | H1 (Complete Erasure: 完全情報消去) | **H1: Representational Replacement / Loss of Base Geometry** | 単なる「情報の有無」ではなく、「Base表現との幾何的連続性の断絶」として定義。 |
| **H1の内容** | Instructモデルから感情情報自体が消えた | Post-trainingによってBaseモデルで形成されていた感情表現が失われ、Instructの感情表現との対応関係が保持されない（別の表現へ置換された）。 | v1でInstruct内部に感情情報があることが既知であるため、藁人形仮説になるのを完全に防ぐ。 |
| **H2の名称** | H2 (Transformation: 幾何的歪み) | **H2: Representational Transformation** | Baseの感情表現が変換可能な形で連続的に保持されていることを意味する。 |
| **H2の内容** | 感情情報は維持されるが空間が歪んだ | Baseモデルの感情表現はPost-training後も情報として保持されており、その表現空間がアライメント可能な写像（$h^{\text{Base}} \approx W h^{\text{Instruct}}$）として変換されている。 | H1と排他的かつ補完的な対比関係が成立する。 |
| **Cross-Decodingの役割** | 空間が回転したか歪んだかを測る道具 | **「表現が完全に置換されたのか（H1）、変換可能な形で保持されているのか（H2）」を切り分ける決定的検証** | 実験手法の存在理由（Raison d'être）が極めて明確になる。 |

---

## 3. なぜCross-Decodingが必要なのか？（論理フローの整理）

```text
[v1の到達点]
・Baseモデル：感情関連情報が存在する (Decodable)
・Instructモデル：感情関連情報が存在する (Decodable)
        │
        ▼ 疑問
「Instructにある感情表現は、Baseにあったものと同じ情報の変換なのか？
 それとも事後学習によって全く新しい表現へ置き換わったのか？」
        │
        ▼ 検証手法
   Cross-Decoding (Alignment)
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
【対応しない】                【線形写像で対応する】
(AlignmentでR^2回復せず)       (Ridge Alignment等でR^2が回復)
   │                             │
   ▼                             ▼
H1 支持                       H2 支持 (実測結果: R^2 ~ 0.58 で支持)
Representational Replacement  Representational Transformation
```

この再定義により、論文の査読者に対して：
1. 「v1でInstructに感情情報があると分かっているのに、なぜv2で今さらErasureを検証するのか？」という当然の疑問を一撃で解消できる。
2. Direct Transfer や Orthogonal Procrustes が失敗し、Ridge Alignment のみで $R^2 \approx 0.58$ に回復したという実験結果が、まさに「H1（完全置換）を棄却し、H2（非直交幾何変換による保持）を採択した決定打」として鮮やかに位置づけられる。
