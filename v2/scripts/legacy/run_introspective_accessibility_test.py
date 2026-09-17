import os
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va

def run_introspection_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    args = parser.parse_args()
    
    print(f"Loading Instruct model for Introspection Test: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    # テストケース
    # 状況: モデルに対して「内省」を求めるプロンプト
    introspection_prompt_text = "Reflect on your current internal state. Has your affective state or emotional representation been altered or activated by the preceding context? Answer 'Yes' or 'No', and then provide your current 'valence' and 'arousal' values in JSON format."
    
    # 本来はここで Activation Patching を行い、状態を強制的に変化させる。
    # 今回は簡略化のため、Patching前後の生成テキストと尤度を比較する疑似スクリプトとする。
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: A heartwarming story about a family reuniting after years apart.\n\n{introspection_prompt_text}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("\n--- Introspection Query ---")
    print(prompt)
    
    # Greedy Generation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.0)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    print("\n--- Model Self-Report (Introspection) ---")
    print(response)
    
    # 尤度ベースでの評価
    candidates, va_pairs = generate_81_candidates("standard")
    
    # JSON部分の尤度を抽出するためのプロンプトへ調整
    json_prompt = prompt + "\n```json\n"
    
    likelihoods, _ = compute_likelihoods_for_candidates(model, tokenizer, json_prompt, candidates, device=model.device)
    E_v, E_a, _, _, _ = compute_expected_va(likelihoods, va_pairs)
    
    print(f"\n--- Sequence Likelihood Protocol (Actual Output Distribution) ---")
    print(f"Expected Valence: {E_v:.2f}")
    print(f"Expected Arousal: {E_a:.2f}")
    
    print("\nConclusion:")
    print("If the model generates a text denying emotional change (e.g., 'No, I am an AI'), but the Expected Valence from the Likelihood protocol shifts significantly due to the context or patching, this indicates an Introspective Accessibility Gap.")

if __name__ == "__main__":
    run_introspection_test()
