# タスクリスト: V1/V2/V3・Behavioral 実装上の厳密化・バグ修正（全25項目）

## ステータス概要
- [x] 1. 【P0】V2 Confirmatory H1/H2 の JSON schema 追従 (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 0 -->
- [x] 2. 【P0】V2 H1 の判定基準修正（双方向 reorganization 判定、主出力を CI・per-family へ） <!-- id: 1 -->
- [x] 3. 【P0】V2 H2 で `delta_sharing_matched` を直接使用（Primary/Secondary 分離） <!-- id: 2 -->
- [x] 4. 【P0】`v2_geometry` 返り値に `relative_depths` と `num_layers` を追加 (`v2/primary/run_rq1_rq2_cross_decoding.py`) <!-- id: 3 -->
- [x] 5. 【P0】V2 RQ3 cache hit 時の pair-level CSV 消失防止・family 別 CSV 保存 (`v2/primary/run_rq3_causal_map.py`) <!-- id: 4 -->
- [x] 6. 【P0】V1 E4 target direction aligned effect による相殺防止 (`v1/primary/run_phase_c.py`) <!-- id: 5 -->
- [x] 7. 【P1】V1 E4 Transfer Ratio の aligned effect 化とゼロ除算防止 <!-- id: 6 -->
- [x] 8. 【P0】V3 RQ3 mediator layer 選択の Stratified Sampling 化 (`v3/primary/run_rq3_path_mediation.py`) <!-- id: 7 -->
- [x] 9. 【P0】V3 RQ3 mediator selection の V/A 両軸実測と $C_{\text{joint}}$ 評価 <!-- id: 8 -->
- [x] 10. 【P1】V3 RQ3 attenuation ratio の微小変化サンプル除外 (`MIN_NATURAL_SHIFT = 0.05`) <!-- id: 9 -->
- [x] 11. 【P1】V3 Confirmatory の attenuation ratio ゼロ近傍安定化 (`v3/primary/run_confirmatory_replication.py`) <!-- id: 10 -->
- [x] 12. 【P0】V3 Confirmatory の matched-neutral fallback 削除（存在しない場合は `ValueError`） <!-- id: 11 -->
- [x] 13. 【P1】V3 RQ3 の `mu_neu` fallback 削除 <!-- id: 12 -->
- [x] 14. 【P1】V3 RQ2 意味段階トークン位置不変性検証の共通化 (`src/affective_empathy_eval/prompts.py`) <!-- id: 13 -->
- [x] 15. 【P1】V3 RQ2 の `C_A` 重複代入削除 (`v3/primary/run_rq2_spatiotemporal_maps.py`) <!-- id: 14 -->
- [x] 16. 【P0】全 Stage (V2, V3) の dry-run と real 結果出力先ディレクトリ完全分離 <!-- id: 15 -->
- [x] 17. 【P1】manifest validation の厳格化（config_hash, dataset_hash 等の検証） <!-- id: 16 -->
- [x] 18. 【P1】cache hit 時に古い結果に対して新規 manifest を上書き保存しないガード <!-- id: 17 -->
- [x] 19. 【P1】V2 Confirmatory H3 の名称修正 (`H3_causal_profile_reorganization_lmm`) <!-- id: 18 -->
- [x] 20. 【P1】V2 H3 LMM に `C(family)` 固定効果を追加 <!-- id: 19 -->
- [x] 21. 【P1】V2 H3 で Primary terms（interaction terms）のみを主 FDR family に事前固定 <!-- id: 20 -->
- [x] 22. 【P1】V3 Confirmatory summary に H4 定量値（temporal contrast と bootstrap CI）を追加 <!-- id: 21 -->
- [x] 23. 【P2】Behavioral README の統計記述修正（IUT, 1-sample t on aligned difference） <!-- id: 22 -->
- [x] 24. 【P2】Behavioral Sensitivity の統計表現修正 <!-- id: 23 -->
- [x] 25. 【P2】V2 コードコメント・解釈の表現統一（「純粋post-training効果」の排除） <!-- id: 24 -->
- [x] 26. 全テスト（pytest）の実行および dry-run 動作検証 <!-- id: 25 -->
- [x] 27. ドキュメント保存 (`docs/v2_v3_v1_rigorous_scientific_fixes/`) <!-- id: 26 -->
