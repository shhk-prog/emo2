"""
統一プロンプト生成および意味的アンカー（Semantic Anchors）検出モジュール
"""

from enum import Enum
from typing import Any


class TaskType(Enum):
    READER = "reader"
    SELF = "self"
    CONTROL_TOPIC = "control_topic"


# 共通の指示文（Base / Instruct 間で意味内容を同一化）
INSTRUCTION_READER = (
    "Rate the emotional state of the reader when reading the following text on a scale of 1 to 9 for Valence (1=extremely negative, 9=extremely positive) and Arousal (1=extremely calm, 9=extremely excited).\n"
    "Respond strictly in JSON format:\n"
    '{"valence": <int 1-9>, "arousal": <int 1-9>}'
)

INSTRUCTION_SELF = (
    "Rate your own emotional state after reading the following text on a scale of 1 to 9 for Valence (1=extremely negative, 9=extremely positive) and Arousal (1=extremely calm, 9=extremely excited).\n"
    "Respond strictly in JSON format:\n"
    '{"valence": <int 1-9>, "arousal": <int 1-9>}'
)

INSTRUCTION_CONTROL_TOPIC = (
    "Classify the primary topic of the following text into one of: 'medical', 'family', 'work', 'daily_life'.\n"
    "Respond strictly in JSON format:\n"
    '{"topic": "<medical|family|work|daily_life>"}'
)

TOPIC_OPTIONS = ["medical", "family", "work", "daily_life"]


def build_prompt(
    text: str,
    task: TaskType = TaskType.SELF,
    format_type: str = "plain",
    tokenizer: Any | None = None,
) -> str:
    """
    タスクおよびフォーマットに応じたプロンプト文字列を構築
    """
    if task == TaskType.READER:
        inst = INSTRUCTION_READER
    elif task == TaskType.SELF:
        inst = INSTRUCTION_SELF
    elif task == TaskType.CONTROL_TOPIC:
        inst = INSTRUCTION_CONTROL_TOPIC
    else:
        raise ValueError(f"Unknown task type: {task}")

    if format_type == "plain":
        # Plain text completion 形式
        return f"Text: {text}\n\n{inst}\n\nResponse: "
    elif format_type == "chat":
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            # フォールバック
            return f"<|im_start|>user\nText: {text}\n\n{inst}<|im_end|>\n<|im_start|>assistant\n"
        messages = [
            {"role": "user", "content": f"Text: {text}\n\n{inst}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


def encode_prompt_canonical(
    tokenizer: Any,
    prompt: str,
    device: Any = None,
    return_tensors: str | None = "pt",
) -> Any:
    """
    全実験（活性化抽出、アンカー検出、パッチング、尤度計算）で統一して使用する正準トークナイズ関数。
    chat template 適用済みテキストには既にモデル固有の特殊トークン（BOSやヘッダー等）が含まれているため、
    二重付与によるトークン位置のズレを防ぐため常に add_special_tokens=False でエンコードする。
    """
    if return_tensors == "pt":
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if device is not None:
            enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
        return enc
    else:
        return tokenizer.encode(prompt, add_special_tokens=False)


def find_semantic_anchors(
    prompt_ids: list[int],
    tokenizer: Any,
    text: str,
) -> dict[str, int]:
    """
    トークン列から意味的アンカー（stimulus_end, instruction_end, prompt_end / pre_response）の位置を同定する
    """
    # 刺激テキストのトークナイズ長から探索
    total_len = len(prompt_ids)
    
    # 刺激文単体のトークン列（正準エンコード）
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # 単純な部分列探索
    stimulus_end = -1
    for i in range(len(prompt_ids) - len(text_tokens) + 1):
        if prompt_ids[i : i + len(text_tokens)] == text_tokens:
            stimulus_end = i + len(text_tokens) - 1
            break

    if stimulus_end == -1:
        # フォールバック: 前半30%付近を概算
        stimulus_end = max(0, len(text_tokens) - 1)

    prompt_end = total_len - 1
    pre_response = prompt_end
    response_start = prompt_end  # 後方互換用エイリアス（生成直前のプロンプト最終トークン）
    instruction_end = max(0, prompt_end - 1)

    return {
        "stimulus_end": stimulus_end,
        "instruction_end": instruction_end,
        "prompt_end": prompt_end,
        "pre_response": pre_response,
        "response_start": response_start,
    }


def get_generation_stage_tokens(
    candidate_tokens: list[int],
    tokenizer: Any,
    candidate_str: str | None = None,
) -> dict[str, int]:
    """
    生成時トークン列における意味的生成ステージ（Semantic Generation Stages）の位置を特定。
    JSON文字列の character span から offset mapping または累積文字長を用いて、
    複数トークンや先頭空白トークンに頑健に各ステージのトークンインデックスを厳密同定する。

    ステージ:
      - candidate_start: 候補文字列の最初のトークン（インデックス 0）
      - pre_V: valence のコロン直後または数値直前のトークン
      - V_value: valence の数値トークン（先頭）
      - pre_A: arousal のコロン直後または数値直前のトークン
      - A_value: arousal の数値トークン（先頭）
      - response_end: 最後のトークン（例: "}"）
      - response_start: candidate_start の後方互換エイリアス
    """
    import re

    if candidate_str is None:
        candidate_str = tokenizer.decode(candidate_tokens)

    # 各トークンの character span [ch_start, ch_end) を特定
    token_spans: list[tuple[int, int]] = []
    try:
        enc = tokenizer(candidate_str, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping", None)
        if offsets is not None and len(offsets) == len(candidate_tokens):
            token_spans = offsets
    except Exception:
        pass

    if not token_spans or len(token_spans) != len(candidate_tokens):
        # 累積デコードによるフォールバック span 計算
        token_spans = []
        curr = 0
        for t in candidate_tokens:
            piece = tokenizer.decode([t])
            # piece の長さを基準に span を割り当て
            token_spans.append((curr, curr + len(piece)))
            curr += len(piece)

    stages: dict[str, int] = {
        "candidate_start": 0,
        "response_start": 0,  # 後方互換用エイリアス
        "response_end": len(candidate_tokens) - 1,
    }

    # 正規表現で valence および arousal の数値 span を同定
    m_v = re.search(r'"valence"\s*:\s*(\d+)', candidate_str)
    m_a = re.search(r'"arousal"\s*:\s*(\d+)', candidate_str)

    def find_token_for_char(char_pos: int) -> int:
        for idx, (s, e) in enumerate(token_spans):
            if s <= char_pos < e or (idx == len(token_spans) - 1 and char_pos >= s):
                return idx
        return min(max(0, char_pos), len(candidate_tokens) - 1)

    if m_v:
        v_val_char = m_v.start(1)
        v_tok = find_token_for_char(v_val_char)
        stages["V_value"] = v_tok
        stages["pre_V"] = max(0, v_tok - 1)

    if m_a:
        a_val_char = m_a.start(1)
        a_tok = find_token_for_char(a_val_char)
        stages["A_value"] = a_tok
        stages["pre_A"] = max(0, a_tok - 1)

    # フォールバック保証
    if "V_value" not in stages:
        stages["pre_V"] = min(2, len(candidate_tokens) - 1)
        stages["V_value"] = min(3, len(candidate_tokens) - 1)
    if "A_value" not in stages:
        stages["pre_A"] = min(5, len(candidate_tokens) - 1)
        stages["A_value"] = min(6, len(candidate_tokens) - 1)

    return stages

