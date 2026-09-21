import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

V3_AFFECTIVE_CONDITIONS = frozenset({"clinical", "affective"})
V3_NEUTRAL_CONDITIONS = frozenset({"neutral"})
V3_CONDITION_COLUMN_CANDIDATES = ("condition", "split")
V3_NEUTRAL_TEXT_COLUMNS = (
    "neutral_text",
    "text_neutral",
    "matched_neutral_text",
    "text_neu",
)


DRY_RUN_VA_LABEL_LOW = 1.0
DRY_RUN_VA_LABEL_HIGH = 9.0


def dry_run_va_label_vector(
    df: pd.DataFrame,
    column: str,
    n: int,
    low: float = DRY_RUN_VA_LABEL_LOW,
    high: float = DRY_RUN_VA_LABEL_HIGH,
) -> np.ndarray:
    """dry-run 専用の決定論的 VA ラベル。列があればそれを使い、無ければ 1..9 の等間隔 fixture。"""
    if column in df.columns:
        return np.asarray(df[column].to_numpy(), dtype=np.float64)
    if n <= 1:
        return np.full(max(n, 0), 0.5 * (low + high), dtype=np.float64)
    return np.linspace(low, high, n, dtype=np.float64)


def describe_loaded_frame(df: pd.DataFrame, name: str, path: Optional[str] = None) -> str:
    """実ロード件数をログ用に整形する。固定件数文字列の代替。"""
    parts = [f"{name}: n_rows={len(df)}"]
    if path:
        parts.append(f"path={path}")
    if "pair_id" in df.columns:
        parts.append(f"n_unique_pair_id={int(df['pair_id'].nunique())}")
    if "stimulus_id" in df.columns:
        parts.append(f"n_unique_stimulus_id={int(df['stimulus_id'].nunique())}")
    if "id" in df.columns and "stimulus_id" not in df.columns:
        parts.append(f"n_unique_id={int(df['id'].nunique())}")
    if "split" in df.columns:
        parts.append(f"split_counts={df['split'].value_counts().to_dict()}")
    return " | ".join(parts)

def scale_vad(raw_value: float) -> float:
    """Scales EmoBank 5-point rating to approximately [-1, 1]."""
    return (raw_value - 3) / 2

def load_emobank(filepath: str, v_col: str = "V", a_col: str = "A") -> pd.DataFrame:
    """Loads EmoBank dataset and applies V/A scaling using explicit column names."""
    df = pd.read_csv(filepath)
    
    if v_col not in df.columns or a_col not in df.columns:
        raise ValueError(f"Columns {v_col} and/or {a_col} not found in {filepath}. Available: {df.columns}")
        
    df['V_scaled'] = df[v_col].apply(scale_vad)
    df['A_scaled'] = df[a_col].apply(scale_vad)
    
    return df

def stratify_stimuli(df: pd.DataFrame, cells_v: int = 3, cells_a: int = 3, n_per_cell: int = 50, seed: int = 42):
    """Stratifies the stimuli into a VA grid and samples n_per_cell.
    Returns: (sampled_df, report_df)
    """
    df_filtered = df.copy()
    
    v_bins = np.linspace(-1, 1, cells_v + 1)
    a_bins = np.linspace(-1, 1, cells_a + 1)
    
    df_filtered['v_cell'] = pd.cut(df_filtered['V_scaled'], bins=v_bins, labels=False, include_lowest=True)
    df_filtered['a_cell'] = pd.cut(df_filtered['A_scaled'], bins=a_bins, labels=False, include_lowest=True)
    
    report_rows = []
    sampled_blocks = []
    
    for (v_cell, a_cell), group in df_filtered.groupby(['v_cell', 'a_cell']):
        candidate_n = len(group)
        sampled_n = min(candidate_n, n_per_cell)
        shortfall_n = n_per_cell - sampled_n
        
        report_rows.append({
            'v_cell': v_cell,
            'a_cell': a_cell,
            'candidate_n': candidate_n,
            'sampled_n': sampled_n,
            'target_n': n_per_cell,
            'shortfall_n': shortfall_n
        })
        
        if sampled_n > 0:
            sampled_blocks.append(group.sample(sampled_n, random_state=seed))
            
    sampled_df = pd.concat(sampled_blocks, ignore_index=True) if sampled_blocks else pd.DataFrame()
    report_df = pd.DataFrame(report_rows)
    
    return sampled_df, report_df

def load_aipsy_affect(cache_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads the AIPsy-Affect dataset from HuggingFace Hub or local cache.
    The dataset provides clinical vignettes with and without emotion keywords/content,
    arranged in minimal pairs.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required to load AIPsy-Affect. Please install it.")
        
    ds_clinical = load_dataset("keidolabs/aipsy-affect", split="clinical", cache_dir=cache_dir)
    ds_neutral = load_dataset("keidolabs/aipsy-affect", split="neutral", cache_dir=cache_dir)
    
    df_clinical = ds_clinical.to_pandas()
    df_clinical["condition"] = "affective"
    
    df_neutral = ds_neutral.to_pandas()
    df_neutral["condition"] = "neutral"
    
    df = pd.concat([df_clinical, df_neutral], ignore_index=True)
    return df

def split_aipsy_affect(df: pd.DataFrame, train_ratio: float = 0.6, dev_ratio: float = 0.2, seed: int = 42):
    """
    Splits the AIPsy-Affect dataset into train/dev/test strictly by Minimal Pair ID
    to prevent data leakage between affective and neutral conditions.
    """
    if "pair_id" not in df.columns:
        if "id" in df.columns:
            df["pair_id"] = df["id"].apply(lambda x: str(x).rsplit("_", 1)[0] if "_" in str(x) else str(x))
        else:
            raise ValueError("Dataset does not contain a 'pair_id' or 'id' column for minimal pair splitting.")
            
    unique_pairs = np.array(df["pair_id"].unique().tolist())
    np.random.seed(seed)
    np.random.shuffle(unique_pairs)
    
    n_pairs = len(unique_pairs)
    n_train = int(n_pairs * train_ratio)
    n_dev = int(n_pairs * dev_ratio)
    
    train_pairs = unique_pairs[:n_train]
    dev_pairs = unique_pairs[n_train:n_train + n_dev]
    test_pairs = unique_pairs[n_train + n_dev:]
    
    train_df = df[df["pair_id"].isin(train_pairs)].copy()
    dev_df = df[df["pair_id"].isin(dev_pairs)].copy()
    test_df = df[df["pair_id"].isin(test_pairs)].copy()
    
    return train_df, dev_df, test_df


def _row_nonempty_text(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def condition_column(df: pd.DataFrame) -> Optional[str]:
    for col in V3_CONDITION_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def resolve_matched_neutral_text(
    row: pd.Series,
    source_df: Optional[pd.DataFrame] = None,
) -> str:
    """同一 pair の中立文を返す。無ければ ValueError（人工中立文は使わない）。"""
    for col in V3_NEUTRAL_TEXT_COLUMNS:
        if col in row.index:
            text = _row_nonempty_text(row[col])
            if text is not None:
                return text

    cond_col = condition_column(source_df) if source_df is not None else None
    row_cond = None
    for col in V3_CONDITION_COLUMN_CANDIDATES:
        if col in row.index:
            row_cond = str(row[col]).strip().lower()
            break
    if row_cond in V3_NEUTRAL_CONDITIONS:
        own = _row_nonempty_text(row.get("text"))
        if own is not None:
            return own

    if source_df is not None and "pair_id" in row.index and "pair_id" in source_df.columns:
        pair_id = row["pair_id"]
        if pd.notna(pair_id):
            pair_rows = source_df[source_df["pair_id"] == pair_id]
            if cond_col is not None:
                neu = pair_rows[pair_rows[cond_col].astype(str).str.lower().isin(V3_NEUTRAL_CONDITIONS)]
            else:
                neu = pair_rows.iloc[0:0]
            if len(neu) > 0:
                text = _row_nonempty_text(neu.iloc[0].get("text"))
                if text is not None:
                    return text

    raise ValueError(
        f"Missing matched-neutral text for stimulus "
        f"{row.get('id', row.get('stimulus_id', 'unknown'))} "
        f"(pair_id={row.get('pair_id', 'missing')}). "
        "V3 Primary forbids a generic neutral sentence or a 5.0 fallback."
    )


def load_v3_matched_pair_table(
    filepath: str,
    require_pairs: bool = True,
) -> pd.DataFrame:
    """V3 Primary 用: 情動側1行 + 同一 pair の neutral_text。

    既定データは AIPsy 4-split の clinical–neutral 192 pair。
    EmoBank 3-way のように pair が無い表は require_pairs=True なら落とす。
    """
    raw = pd.read_csv(filepath)
    cond_col = condition_column(raw)
    if cond_col is None or "pair_id" not in raw.columns:
        if require_pairs:
            raise ValueError(
                f"V3 Primary requires pair_id and a condition/split column with matched neutrals. "
                f"Got columns={list(raw.columns)} from {filepath}."
            )
        out = raw.copy()
        out["neutral_text"] = out.apply(lambda r: resolve_matched_neutral_text(r, raw), axis=1)
        return out.reset_index(drop=True)

    cond = raw[cond_col].astype(str).str.lower()
    aff = raw[cond.isin(V3_AFFECTIVE_CONDITIONS) & raw["pair_id"].notna()].copy()
    records = []
    skipped = 0
    skipped_records = []
    for _, row in aff.iterrows():
        try:
            neu = resolve_matched_neutral_text(row, raw)
        except ValueError as e:
            skipped += 1
            skipped_records.append({
                "pair_id": row.get("pair_id", "unknown"),
                "reason": str(e),
                "text": str(row.get("text", ""))[:100],
            })
            continue
        rec = row.to_dict()
        rec["condition"] = "affective"
        rec["neutral_text"] = neu
        records.append(rec)

    if skipped_records:
        # AGENTS.md 1.2 / Audit Item 38: 除外理由を明示的に exclusions_v3.csv / v3_exclusions.csv に保存
        for ex_fname in ["exclusions_v3.csv", "v3_exclusions.csv"]:
            ex_path = Path("v3/results/derived") / ex_fname
            try:
                ex_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(skipped_records).to_csv(ex_path, index=False)
            except Exception as e:
                logger.warning(f"Failed to save V3 exclusions to {ex_path}: {e}")

    if not records:
        raise ValueError(
            f"No matched affective–neutral pairs could be built from {filepath}."
        )
    out = pd.DataFrame(records).reset_index(drop=True)
    out.attrs["n_input_pairs"] = int(len(aff))
    out.attrs["n_matched_pairs"] = int(len(records))
    out.attrs["n_excluded_pairs"] = int(skipped)
    out.attrs["n_skipped_unmatched"] = int(skipped)
    return out


def stratified_causal_subset(
    df: pd.DataFrame,
    n_samples: int,
    seed: int = 42,
    stratify_col: str = "target_emotion",
) -> List[int]:
    """
    感情カテゴリ等から層化サンプリングしたインデックスリストを返す。
    まず各カテゴリから1件ずつ均等に抽出し、残りをシード付きランダムで補充する。
    """
    N = len(df)
    n_target = min(n_samples, N)
    rng = np.random.default_rng(seed)

    selected_indices: List[int] = []
    # 候補列の確認 (target_emotion または emotion)
    col = stratify_col if stratify_col in df.columns else ("emotion" if "emotion" in df.columns else None)
    if col is not None:
        for _, grp in df.groupby(col):
            if len(grp) > 0:
                idx = int(rng.choice(grp.index.to_numpy(), size=1)[0])
                selected_indices.append(idx)

    # 上限を超える場合は削る
    if len(selected_indices) > n_target:
        selected_indices = sorted(rng.choice(selected_indices, size=n_target, replace=False).tolist())
    else:
        # 不足分をシード付きランダムで均等補充
        rem_needed = n_target - len(selected_indices)
        if rem_needed > 0:
            remaining = [i for i in range(N) if i not in selected_indices]
            if remaining:
                supp = rng.choice(remaining, size=min(rem_needed, len(remaining)), replace=False).tolist()
                selected_indices.extend(supp)

    return sorted(selected_indices)


def v3_stimulus_covariate(df: pd.DataFrame, axis: str, n: Optional[int] = None) -> np.ndarray:
    """
    β 用の刺激共変量行列。
    人間 VA があれば連続ラベルを使用し、無ければ target_emotion, intensity, domain を
    ダミー変数化 (one-hot, drop_first=True) してカテゴリ共変量として統制する。
    """
    n_rows = n if n is not None else len(df)
    if axis == "v" and "reader_V" in df.columns:
        return np.asarray(df["reader_V"].to_numpy(), dtype=np.float64).reshape(-1, 1)
    if axis == "a" and "reader_A" in df.columns:
        return np.asarray(df["reader_A"].to_numpy(), dtype=np.float64).reshape(-1, 1)

    # AIPsy 等で人間連続 VA が無い場合: target_emotion (または emotion) / intensity / domain を one-hot 化
    covar_cols = []
    if "target_emotion" in df.columns:
        covar_cols.append("target_emotion")
    elif "emotion" in df.columns:
        covar_cols.append("emotion")
    for c in ["intensity", "domain"]:
        if c in df.columns:
            covar_cols.append(c)

    if covar_cols:
        dummies = pd.get_dummies(df[covar_cols].astype(str), drop_first=True, dtype=float)
        if dummies.shape[1] > 0:
            return dummies.values[:n_rows]

    return np.zeros((n_rows, 1), dtype=np.float64)


# Backward-compatibility aliases
load_emobank_csv = load_emobank
load_aipsy_csv = load_aipsy_affect

