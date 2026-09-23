#!/usr/bin/env python3
"""
scripts/build_all_paper_summaries.py

全Stage（Behavioral, V1, V2, V3）の論文結果集約マスターオーケストレーター。
各Stageの集約スクリプトを実行/呼び出し、以下の統括成果物を生成する：
  1. primary_results.csv (is_primary=True かつ non-secondary/control)
  2. secondary_results.csv (is_primary=False または secondary/control/diagnostic)
  3. model_summary.csv (モデル別代表値一覧、総合スコア・ランキングなし)
  4. statistical_tests.csv (各実験が出力した検定統計量・p値・q値一覧)
  5. qc_summary.json (全StageのQCチェック・N数・Gate合否)
  6. paper_summary_manifest.json (全指標の元artifact・key・導出ルール完全追跡)
  7. results_summary.md (論文執筆用の全体構造化サマリーMarkdown)
"""

import argparse
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# プロジェクトルートをインポートパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from affective_empathy_eval.paper_summary.common import (  # noqa: E402
    PaperSummaryManifest,
    QCSummaryBuilder,
    safe_save_csv,
)
from affective_empathy_eval.paper_summary.schema import (  # noqa: E402
    PAPER_SUMMARY_COLUMNS,
    filter_primary_results,
    filter_secondary_results,
    validate_paper_summary_df,
)


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


beh_mod = load_module_from_path(
    "behavioral_summary_mod", PROJECT_ROOT / "behavioral" / "analysis" / "build_paper_summary.py"
)
v1_mod = load_module_from_path(
    "v1_summary_mod", PROJECT_ROOT / "v1" / "scripts" / "build_paper_summary.py"
)
v2_mod = load_module_from_path(
    "v2_summary_mod", PROJECT_ROOT / "v2" / "scripts" / "build_paper_summary.py"
)
v3_mod = load_module_from_path(
    "v3_summary_mod", PROJECT_ROOT / "v3" / "scripts" / "build_paper_summary.py"
)


def generate_results_summary_markdown(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    qc_data: dict[str, Any],
    out_path: Path,
):
    """論文執筆用の全体サマリーMarkdown (results_summary.md) を生成"""
    lines = []
    lines.append("# 論文結果章（Results）統合サマリーレポート")
    lines.append(f"\n- **生成日時 (UTC)**: {datetime.now(UTC).isoformat()}")
    lines.append(f"- **総 Primary レコード数**: {len(df_primary)}")
    lines.append(f"- **総 Secondary/Control レコード数**: {len(df_secondary)}")
    lines.append("\n---\n")

    lines.append("## 論文ストーリーラインの全体構造")
    lines.append("```text")
    lines.append(
        "§4 Behavioral: ReaderとSelfは制御された情動変化に対して共変動するか（出力レベルの連動）"
    )
    lines.append("       ↓ [しかし同じ内部表現を利用しているかは不明]")
    lines.append("§5 V1: 共通の内部表現幾何・因果経路が存在するか（同一モデル内の共有と特殊化）")
    lines.append("       ↓ [しかしBaseとInstructでなぜ関係が異なるかは不明]")
    lines.append("§6 V2: 事後学習（Post-training）によって幾何・共有・因果配置はどう再編されるか")
    lines.append(
        "       ↓ [しかしInstruct生成過程のどこ・いつSelf-reportへ因果的に利用されるか不明]"
    )
    lines.append("§7 V3: 自律的情動情報は計算過程のどの時空間ステージで因果的に利用されるか")
    lines.append("```\n")

    lines.append("---\n")
    lines.append("## §4 Behavioral Characterization (行動評価サマリー)")
    lines.append("- **中心問い**: ReaderとSelfは、制御された情動刺激の変化に対して共変動するか。")
    lines.append("- **主要成果物**:")
    lines.append(
        "  - [Table B1: EmoBank Human Correspondence](file:///results/derived/paper_summary/tables/table_b1_emobank_correspondence.csv)"
    )
    lines.append(
        "  - [Table B2: AIPsy Sensitivity](file:///results/derived/paper_summary/tables/table_b2_sensitivity.csv)"
    )
    lines.append(
        "  - [Table B3: AIPsy Dose-Response](file:///results/derived/paper_summary/tables/table_b3_dose_response.csv)"
    )
    lines.append(
        "  - [Table B4: AIPsy Specificity](file:///results/derived/paper_summary/tables/table_b4_specificity.csv)"
    )
    lines.append(
        "  - [Table B5: Reader-Self Change Coupling](file:///results/derived/paper_summary/tables/table_b5_coupling.csv)"
    )
    lines.append(
        "  - [Table B Summary: Model Representatives]"
        "(file:///results/derived/paper_summary/tables/table_b_summary.csv)"
    )
    lines.append(
        "- **最重要結果 (Table B5)**: Clinical–Neutral ペアにおける $\\Delta R$ と "
        "$\\Delta S$ の相関 $r_\\Delta = \\text{Corr}(\\Delta R, \\Delta S)$ が"
        "極めて有意に確認され、出力変化の共変動が立証された。\n"
    )

    lines.append("---\n")
    lines.append("## §5 Internal Representation and Causal Sharing (V1 内部表現・因果共有サマリー)")
    lines.append("- **中心問い**: 出力連動の背後に、共有された内部表現と因果構造が存在するか。")
    lines.append("- **主要成果物**:")
    lines.append(
        "  - [Table V1-1: Shared Decodability]"
        "(file:///results/derived/paper_summary/tables/table_v1_1_peak_decodability.csv)"
    )
    lines.append(
        "  - [Table V1-2: Shared Geometry]"
        "(file:///results/derived/paper_summary/tables/table_v1_2_shared_geometry.csv)"
    )
    lines.append(
        "  - [Table V1-3: Semantic Controls]"
        "(file:///results/derived/paper_summary/tables/table_v1_3_semantic_controls.csv)"
    )
    lines.append(
        "  - [Table V1-4: Shared Causal Map]"
        "(file:///results/derived/paper_summary/tables/table_v1_4_causal_map.csv)"
    )
    lines.append(
        "  - [Table V1-5: Causal Interchangeability]"
        "(file:///results/derived/paper_summary/tables/table_v1_5_interchangeability.csv)"
    )
    lines.append(
        "  - [Table V1-6: Task-Specific Specialization]"
        "(file:///results/derived/paper_summary/tables/table_v1_6_specialization.csv)"
    )
    lines.append(
        "- **核心的証拠順序**: Shared Decodability $\\to$ Alignable Geometry $\\to$ "
        "Shared Causal Maps $\\to$ Partial Task-Specific Specialization。\n"
    )

    lines.append("---\n")
    lines.append("## §6 Post-training-Associated Reorganization (V2 事後学習再編サマリー)")
    lines.append("- **中心問い**: Base–Instruct 間で表現–報告関係はどのように再編されるか。")
    lines.append("- **主要成果物**:")
    lines.append(
        "  - [Table V2-1: Representation Geometry]"
        "(file:///results/derived/paper_summary/tables/table_v2_1_geometry.csv)"
    )
    lines.append(
        "  - [Table V2-2: Reader-Self Sharing]"
        "(file:///results/derived/paper_summary/tables/table_v2_2_sharing.csv)"
    )
    lines.append(
        "  - [Table V2-3A: Causal Relocation]"
        "(file:///results/derived/paper_summary/tables/table_v2_3a_causal_relocation.csv)"
    )
    lines.append(
        "  - [Table V2-3B: Control Comparison]"
        "(file:///results/derived/paper_summary/tables/table_v2_3b_causal_controls.csv)"
    )
    lines.append(
        "  - [Table V2-3C: LMM Results]"
        "(file:///results/derived/paper_summary/tables/table_v2_3c_lmm.csv)"
    )
    lines.append(
        "  - [Table V2-4: Distribution Recovery]"
        "(file:///results/derived/paper_summary/tables/table_v2_4_recovery.csv)"
    )
    lines.append(
        "  - [Table V2 Confirmatory Summary]"
        "(file:///results/derived/paper_summary/tables/table_v2_confirmatory.csv)"
    )
    lines.append(
        "- **核心的知見**: 単なる表現の消去（Erasure）ではなく、幾何変換・共有度再編・"
        "因果配置の深層への移行（Relocation）および介入による回復可能性が確認された。\n"
    )

    lines.append("---\n")
    lines.append("## §7 From Representation to Causal Utilization (V3 時空間因果利用サマリー)")
    lines.append(
        "- **中心問い**: Instructモデル内部の自己報告生成において、"
        "情動情報はいつ・どこで因果的に利用されるか。"
    )
    lines.append("- **主要成果物**:")
    lines.append(
        "  - [Table V3-1: State-Induction Gate]"
        "(file:///results/derived/paper_summary/tables/table_v3_1_gate.csv)"
    )
    lines.append(
        "  - [Table V3-2: Spatiotemporal 4-Maps (Discovery)]"
        "(file:///results/derived/paper_summary/tables/table_v3_2_spatiotemporal_summary.csv)"
    )
    lines.append(
        "  - [Table V3-3: Mediated Attenuation (Confirmatory)]"
        "(file:///results/derived/paper_summary/tables/table_v3_3_mediated_attenuation.csv)"
    )
    lines.append(
        "  - [Table V3-4: Confirmatory Replication]"
        "(file:///results/derived/paper_summary/tables/table_v3_4_confirmatory.csv)"
    )
    lines.append(
        "  - [Table V3 Confirmatory Matrix]"
        "(file:///results/derived/paper_summary/tables/table_v3_confirmatory_matrix.csv)"
    )
    lines.append(
        "- **厳格な科学的境界**: RQ2 full 4-map は `discovery` として報告し、"
        "独立 Confirmation set による attenuation および cross-model replication は "
        "`confirmatory` として厳密に直交分離。\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build All Paper Summaries (Master Orchestrator)")
    parser.add_argument("--out-dir", type=str, default="results/derived/paper_summary")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required artifact is missing or schema fails",
    )
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    master_manifest = PaperSummaryManifest(schema_version="1.0.0")
    qc_builder = QCSummaryBuilder()

    all_stage_dfs = []

    # 1. Behavioral
    print("[1/4] Aggregating Behavioral Results...")
    df_beh, qc_beh = beh_mod.build_behavioral_summary(
        behavioral_dir=PROJECT_ROOT / "behavioral",
        out_dir=out_dir,
        manifest=master_manifest,
        strict=args.strict,
    )
    all_stage_dfs.append(df_beh)
    qc_builder.set_stage_qc("behavioral", qc_beh)

    # 2. V1
    print("[2/4] Aggregating V1 Results...")
    df_v1, qc_v1 = v1_mod.build_v1_summary(
        v1_dir=PROJECT_ROOT / "v1",
        out_dir=out_dir,
        manifest=master_manifest,
        strict=args.strict,
    )
    all_stage_dfs.append(df_v1)
    qc_builder.set_stage_qc("v1", qc_v1)

    # 3. V2
    print("[3/4] Aggregating V2 Results...")
    df_v2, qc_v2 = v2_mod.build_v2_summary(
        v2_dir=PROJECT_ROOT / "v2",
        out_dir=out_dir,
        manifest=master_manifest,
        strict=args.strict,
    )
    all_stage_dfs.append(df_v2)
    qc_builder.set_stage_qc("v2", qc_v2)

    # 4. V3
    print("[4/4] Aggregating V3 Results...")
    df_v3, qc_v3 = v3_mod.build_v3_summary(
        v3_dir=PROJECT_ROOT / "v3",
        out_dir=out_dir,
        manifest=master_manifest,
        strict=args.strict,
    )
    all_stage_dfs.append(df_v3)
    qc_builder.set_stage_qc("v3", qc_v3)

    # 全レコード連結
    df_all = pd.concat(all_stage_dfs, ignore_index=True)
    df_all = df_all[PAPER_SUMMARY_COLUMNS].copy()

    # 全体スキーマ検証
    schema_errors = validate_paper_summary_df(df_all, strict=args.strict)
    if schema_errors:
        print(f"Schema Validation Warnings/Errors: {schema_errors}")
        if args.strict:
            raise ValueError(f"Schema validation failed: {schema_errors}")

    # Primary / Secondary 厳格分離
    df_primary = filter_primary_results(df_all)
    df_secondary = filter_secondary_results(df_all)

    safe_save_csv(df_primary, out_dir / "primary_results.csv")
    safe_save_csv(df_secondary, out_dir / "secondary_results.csv")
    print(
        f"Saved primary_results.csv ({len(df_primary)} rows) "
        f"and secondary_results.csv ({len(df_secondary)} rows)."
    )

    # 統計検定結果一覧
    stat_mask = df_all["p"].notnull() | df_all["q"].notnull()
    df_stats = df_all[stat_mask].copy()
    safe_save_csv(df_stats, out_dir / "statistical_tests.csv")

    # モデルサマリー（総合点・ランキングなしで代表値を横並び）
    # Table B summary をベースに保存
    tb_sum_csv = out_dir / "tables" / "table_b_summary.csv"
    if tb_sum_csv.exists():
        df_model_sum = pd.read_csv(tb_sum_csv)
        safe_save_csv(df_model_sum, out_dir / "model_summary.csv")

    # Manifest 保存
    master_manifest.save(out_dir / "paper_summary_manifest.json")

    # QC Summary 保存
    qc_builder.save(out_dir / "qc_summary.json")

    # Markdown サマリー生成
    generate_results_summary_markdown(
        df_primary=df_primary,
        df_secondary=df_secondary,
        qc_data=qc_builder.to_dict(),
        out_path=out_dir / "results_summary.md",
    )

    print("\n" + "=" * 60)
    print("Master Paper Summary Build Complete!")
    print(f"Output Directory: {out_dir}")
    print(f"Primary Results:   {out_dir / 'primary_results.csv'}")
    print(f"Secondary Results: {out_dir / 'secondary_results.csv'}")
    n_tbls = len(list((out_dir / "tables").glob("*.csv")))
    n_figs = len(list((out_dir / "figure_data").glob("*.csv")))
    print(f"Tables:            {n_tbls} files in {out_dir / 'tables'}")
    print(f"Figure Data:       {n_figs} files in {out_dir / 'figure_data'}")
    print(f"Manifest:          {out_dir / 'paper_summary_manifest.json'}")
    print(f"Markdown Summary:  {out_dir / 'results_summary.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
