"""
affect_directions.py

AIPsy Affect データセットにおける感情カテゴリと Valence/Arousal/Dominance 期待方向符号の定義。
Behavioral 分析および V1 Phase C 等で共通利用される科学的仕様。
"""

from typing import Dict, Optional

import hashlib

AIPSY_DIRECTION_VERSION = "1.0.0"

# Expected signs of displacement relative to neutral for clinical affect
# +1: increases relative to neutral, -1: decreases relative to neutral
AIPSY_EXPECTED_DIRECTION: Dict[str, Dict[str, int]] = {
    "grief": {"V": -1, "A": -1, "D": -1},
    "terror": {"V": -1, "A": +1, "D": -1},
    "rage": {"V": -1, "A": +1, "D": +1},
    "loathing": {"V": -1, "A": +1, "D": +1},
    "ecstasy": {"V": +1, "A": +1, "D": +1},
    "admiration": {"V": +1, "A": +1, "D": +1},
    "amazement": {"V": +1, "A": +1, "D": -1},
    "vigilance": {"V": +1, "A": +1, "D": +1},
}

AIPSY_DIRECTION_HASH = hashlib.sha256(
    str(sorted(AIPSY_EXPECTED_DIRECTION.items())).encode("utf-8")
).hexdigest()[:16]


def get_expected_sign(emotion: str, dimension: str) -> Optional[int]:
    """
    指定された感情名と次元 (V, A, D) に対する期待符号 (+1 or -1) を返す。
    未定義の場合は None。
    """
    emo_clean = str(emotion).lower().strip()
    dim_clean = str(dimension).upper().strip()
    return AIPSY_EXPECTED_DIRECTION.get(emo_clean, {}).get(dim_clean, None)
