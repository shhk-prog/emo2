import hashlib
import json
from transformers import AutoTokenizer

def apply_chat_template_val_forcing(tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += '{\n  "valence": '
    return prompt

def main():
    target_model = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading tokenizer: {target_model}")
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    
    # 1. Base prompt generation
    placeholder_text = "<INSERT_TEXT_HERE>"
    prompt = apply_chat_template_val_forcing(tokenizer, placeholder_text)
    
    # 2. Hash calculation
    sha256_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    
    output_data = {
        "model": target_model,
        "template_name": "strict_val_forcing",
        "placeholder_text": placeholder_text,
        "exact_prompt_string": prompt,
        "sha256": sha256_hash
    }
    
    with open("v2/results/prompt_hashes.json", "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved prompt hashes to v2/results/prompt_hashes.json")
    print(f"SHA-256: {sha256_hash}")

if __name__ == "__main__":
    main()
