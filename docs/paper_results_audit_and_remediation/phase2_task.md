# Phase 2: 結果生成ロジック及びLaTeXブロッカーの完全解消タスク

## 目的
ユーザーからの最新フィードバックに基づき、残る15項目のブロッカー（V2生成ロジック・V3判定ロジック・テーブル整合性・結果ファイル配置）およびLaTeXコンパイルエラー（`Missing } inserted`）を完全に解消する。

## タスクリスト

### 1. V2 再編成パイプラインの修正 (項目 1, 2, 3, 4, 5, 6, 7, 8)
- [ ] 1. **V2 recovery tableの実結果再生成**: `v2/scripts/build_paper_summary.py` にて `v2_distribution_recovery_summary.json` から全4ファミリー (gemma, llama, olmo, qwen) の Reader / Self 実データを読み込み、`table_v2_4_recovery.csv` を出力。
- [ ] 2. **V2 H1/H2 hard-coded fallback削除**: `scripts/summarize_v2_reorganization.py` の `generate_h1_h2_table()` 内のハードコードデフォルト値を廃止。データ欠損時は例外または `---` に。
- [ ] 3. **V2 confirmatory H3/H4追加**: `v2_confirmatory_summary.tex` に H3 (Causal Profile Reorganization) と H4 (Distribution Recovery) の行を追加し、caption（H1--H4）と完全一致させる。
- [ ] 4. **V2 confirmatory q値ハードコード削除**: `< 0.001` の一律出力を廃止し、実データのq値または `---` を出力。
- [ ] 5. **V2 causal centerの真のcenter-of-mass反映**: $\bar{d}_C = \sum_l d_l \max(C(l),0) / \sum_l \max(C(l),0)$ の計算結果をテーブル出力へ反映。
- [ ] 6. **V2 causal controlsにTask列追加**: `v2_causal_controls.tex` に `Task` 列を追加し、Reader / Self の行の重複・潰れを解消。
- [ ] 7. **V2 control tableのNote修正**: $C_{\mathrm{net,rand}}>0$ が統計的有意性を示すわけではない旨の注記に改訂。
- [ ] 8. **V2 LMM caption修正**: 目的変数が $C_{\mathrm{net,rand}}$ である旨を明記。

### 2. V3 因果利用パイプラインの修正 (項目 9, 10, 11, 12, 13, 14, 15, 16)
- [ ] 9. **V3 H1 符号判定をMethods通り正方向へ修正**: $\Delta d^* = d_C^* - d_D^* > 0$ かつ $\Delta \bar{d} > 0$（因果ピーク/中心がデコードより深層）。
- [ ] 10. **V3 H1 判定にcenter criterion追加**: peakとcenterの双方のCI下限が0超過を要求。
- [ ] 11. **V3 H3 閾値修正**: $0.05$ ではなく $CI_{\mathrm{low}}(M) > 0$ かつ $CI_{\mathrm{low}}(M_{\mathrm{net}}) > 0$。
- [ ] 12. **V3 H4 閾値修正**: $0.05$ ではなく $CI_{\mathrm{low}}(\Delta C^{\mathrm{temporal}}) > 0$。
- [ ] 13. **V3 mediated attenuationのハードコード削除**: `m_rand = "0.0000"` や逆算を廃止し、実結果の $T, R, M, M_{\mathrm{rand}}, M_{\mathrm{net}}$ を `table_v3_3_mediated_attenuation.csv` に直接反映。
- [ ] 14. **V3 canonical builderとpresentation summarizerの一致**: `v3/scripts/build_paper_summary.py` を2軸対応・厳密判定へ更新し、`scripts/summarize_v3_causal_utilization.py` と同期。
- [ ] 15. **V3 Noteの整合性修正**: H4 Arousal単軸passと二軸joint confirmationの不達成を正確に記述。

### 3. データ集約・配置 (項目 15/18)
- [ ] 16. `results/derived/paper_summary/` 配下に全最新CSV・要約・マニフェストを出力・配置。

### 4. LaTeX監査・コンパイル確認 (項目 19)
- [ ] 17. `iclr2027/iclr2027_conference2.tex` 内の `Missing } inserted` 等のエラーを特定・解消。
- [ ] 18. LaTeX全体のビルド検証。
