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
    response_start = prompt_end  # 後方互換用エイリアス（実際には生成直前のプロンプト最終トークン）
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
) -> dict[str, int]:
    """
    生成時トークン列における意味的生成ステージ（Semantic Generation Stages）の位置を特定
    ステージ:
      - response_start: 最初のトークン
      - pre_V: valence の値の直前のトークン（例: ":"）
      - V_value: valence の数値トークン
      - pre_A: arousal の値の直前のトークン（例: ":"）
      - A_value: arousal の数値トークン
      - response_end: 最後のトークン（例: "}"）
    """
    # デコードしてテキスト内位置からトークン対応を求める
    token_strs = [tokenizer.decode([t]) for t in candidate_tokens]
    
    stages: dict[str, int] = {}
    stages["response_start"] = 0
    stages["response_end"] = len(candidate_tokens) - 1

    # valence / arousal 数値の探索
    for i, t_str in enumerate(token_strs):
        if any(char.isdigit() for char in t_str):
            if "V_value" not in stages:
                stages["V_value"] = i
                stages["pre_V"] = max(0, i - 1)
            elif "A_value" not in stages:
                stages["A_value"] = i
                stages["pre_A"] = max(0, i - 1)

    # フォールバック保証
    if "V_value" not in stages:
        stages["pre_V"] = min(2, len(candidate_tokens) - 1)
        stages["V_value"] = min(3, len(candidate_tokens) - 1)
    if "A_value" not in stages:
        stages["pre_A"] = min(5, len(candidate_tokens) - 1)
        stages["A_value"] = min(6, len(candidate_tokens) - 1)

    return stages
