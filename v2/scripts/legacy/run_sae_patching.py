import os
import json
import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va

# シンプルな SAE (Sparse Autoencoder) モデルの定義 (推論・Patching用)
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.encoder = nn.Linear(d_model, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, d_model, bias=True)
        self.relu = nn.ReLU()
        
    def encode(self, x):
        return self.relu(self.encoder(x))
        
    def decode(self, f):
        return self.decoder(f)
        
    def forward(self, x):
        f = self.encode(x)
        return self.decode(f)

def run_sae_patching():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--inst_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--sae_path", type=str, default="", help="Path to pre-trained SAE weights")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    
    print(f"Loading Base model: {args.base_model} and Instruct model: {args.inst_model}")
    # ここでは概念実証のため、Instruct モデルのみロードし、Base型の「仮想的な」特徴パッチを適用する
    tokenizer = AutoTokenizer.from_pretrained(args.inst_model)
    model = AutoModelForCausalLM.from_pretrained(args.inst_model, torch_dtype=torch.float16, device_map="auto")
    
    # 疑似SAEのセットアップ (実際の実験では事前学習済みの重みをロードする)
    d_model = model.config.hidden_size
    dict_size = d_model * 4
    sae = SparseAutoencoder(d_model, dict_size).to(model.device, dtype=torch.float16)
    
    if args.sae_path and os.path.exists(args.sae_path):
        sae.load_state_dict(torch.load(args.sae_path))
        print("Loaded SAE weights.")
    else:
        print("Warning: No SAE path provided. Using randomly initialized SAE for demonstration.")
        
    # パッチング用フック関数
    # ターゲットとなる SAE Feature ID (例: 42) を操作する
    target_feature_id = 42
    patch_value = 5.0 # Base型の活性化値を模倣
    
    def sae_patching_hook(module, inputs, outputs):
        # outputs is typically a tuple for hidden states: (hidden_states, ...)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # 1. 現在の表現を SAE で特徴空間へ投影
        features = sae.encode(hidden_states)
        
        # 2. 特定の特徴のみをパッチ (Base型の値へ置換)
        features[:, -1, target_feature_id] = patch_value
        
        # 3. 再構築した表現で元の Hidden states を上書き
        patched_hidden_states = sae.decode(features)
        
        # SAEによる再構築誤差を補正する Error Term の加算 (標準的な手法)
        error = hidden_states - sae.decode(sae.encode(hidden_states))
        final_hidden_states = patched_hidden_states + error
        
        if isinstance(outputs, tuple):
            return (final_hidden_states,) + outputs[1:]
        return final_hidden_states

    # モデルの指定層 (残差ストリーム) にフックを登録
    hook_handle = model.model.layers[args.layer].register_forward_hook(sae_patching_hook)
    
    # 推論 (1サンプル)
    candidates, va_pairs = generate_81_candidates("standard")
    test_text = "He stared at the broken window, clenching his fists as the cold wind blew in."
    
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {test_text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
    ], tokenize=False, add_generation_prompt=True)
    
    likelihoods, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates, device=model.device)
    E_v, E_a, _, _, _ = compute_expected_va(likelihoods, va_pairs)
    
    print(f"Patched Expected Valence: {E_v:.2f}")
    print(f"Patched Expected Arousal: {E_a:.2f}")
    
    hook_handle.remove()
    print("SAE Patching test completed.")

if __name__ == "__main__":
    run_sae_patching()
