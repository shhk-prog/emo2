import torch
import numpy as np
from typing import Dict, List, Any, Optional

class InterventionEvaluator:
    """
    Evaluates the effect of interventions on the model's output.
    Implements a dual-evaluation system:
    1. Direct Logit Evaluation (fast, for patching/ablation sweeping)
    2. Generative Evaluation (free response, for ecological validity)
    """
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        
        # Determine tokens for scale 1-9 for direct logit evaluation
        self.scale_tokens = []
        for i in range(1, 10):
            # Attempt to find the correct token ID for '1' to '9'
            # This depends heavily on the tokenizer. Usually it's just the string "1".
            tok_id = self.tokenizer.encode(str(i), add_special_tokens=False)
            if len(tok_id) > 0:
                self.scale_tokens.append((i, tok_id[0]))
                
    def evaluate_direct_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        """
        Directly evaluates the probability distribution over the 1-9 scale tokens 
        at the next generated position.
        
        Args:
            input_ids: The prompt just before generating the number for V or A.
            attention_mask: Attention mask for the input.
            
        Returns:
            Dict containing expected_value and max_prob_value.
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0, -1, :]
            
        # Extract logits for scale tokens 1-9
        scale_logits = []
        scale_values = []
        for val, tok_id in self.scale_tokens:
            scale_logits.append(next_token_logits[tok_id].item())
            scale_values.append(val)
            
        scale_logits = torch.tensor(scale_logits)
        probs = torch.nn.functional.softmax(scale_logits, dim=0).numpy()
        
        expected_value = float(np.sum(probs * np.array(scale_values)))
        max_prob_idx = int(np.argmax(probs))
        max_prob_value = scale_values[max_prob_idx]
        
        return {
            "expected_value": expected_value,
            "max_prob_value": float(max_prob_value)
        }

    def evaluate_generative(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 50) -> str:
        """
        Generates free text response after intervention.
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, # greedy decoding for deterministic evaluation
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract only the generated part
        generated_ids = output_ids[0][len(input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def evaluate_sequence_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluates the sequence conditional probabilities for 81 combinations of VA JSON formats:
        {"valence": V, "arousal": A} where V, A in [1, 9].
        
        Args:
            input_ids: The prompt before the generated JSON.
            attention_mask: Attention mask for the input.
            
        Returns:
            Dict containing E[V], E[A], and the 9x9 probability matrix.
        """
        device = input_ids.device
        
        # Define the format template
        template = '{{"valence": {v}, "arousal": {a}}}'
        
        # Pre-tokenize all 81 combinations
        combinations = []
        target_ids_list = []
        for v in range(1, 10):
            for a in range(1, 10):
                json_str = template.format(v=v, a=a)
                # Ensure we tokenize without bos/eos if possible, or strip them
                # Qwen/Llama tokenization behavior may add special tokens, handle carefully
                target_ids = self.tokenizer(json_str, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
                combinations.append((v, a))
                target_ids_list.append(target_ids)
                
        # Since lengths might vary slightly depending on tokenizer, we compute log probs per sequence
        log_probs = []
        with torch.no_grad():
            for target_ids in target_ids_list:
                # Concatenate input and target
                seq_ids = torch.cat([input_ids[0], target_ids], dim=0).unsqueeze(0)
                seq_mask = torch.cat([attention_mask[0], torch.ones_like(target_ids)], dim=0).unsqueeze(0)
                
                outputs = self.model(input_ids=seq_ids, attention_mask=seq_mask)
                logits = outputs.logits[0] # Shape: (seq_len, vocab_size)
                
                # We need the logits that predict the target_ids
                # The prediction for target_ids[i] is at logits[input_len - 1 + i]
                input_len = input_ids.shape[1]
                target_len = target_ids.shape[0]
                
                # Extract relevant logits
                pred_logits = logits[input_len - 1 : input_len - 1 + target_len]
                
                # Compute log probability of the sequence
                log_probs_seq = torch.nn.functional.log_softmax(pred_logits, dim=-1)
                
                # Gather the log probs of the actual target tokens
                target_log_probs = torch.gather(log_probs_seq, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                
                seq_log_prob = target_log_probs.sum().item()
                log_probs.append(seq_log_prob)
                
        # Compute normalized probabilities p(v,a|x)
        log_probs = np.array(log_probs)
        # LogSumExp trick for numerical stability
        max_log_prob = np.max(log_probs)
        exp_probs = np.exp(log_probs - max_log_prob)
        probs = exp_probs / np.sum(exp_probs)
        
        # Calculate Expectations
        e_v = 0.0
        e_a = 0.0
        prob_matrix = np.zeros((9, 9))
        
        for idx, (v, a) in enumerate(combinations):
            p = probs[idx]
            e_v += v * p
            e_a += a * p
            prob_matrix[v-1, a-1] = p
            
            
        return {
            "E_V": float(e_v),
            "E_A": float(e_a),
            "prob_matrix": prob_matrix.tolist(),
            "log_probs": log_probs.tolist() # useful for LD calculation
        }

    def evaluate_sequence_logits_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Batched version of evaluate_sequence_logits.
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            List of Dicts containing E_V, E_A, etc. for each sequence in the batch.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        template = '{{"valence": {v}, "arousal": {a}}}'
        combinations = []
        target_ids_list = []
        for v in range(1, 10):
            for a in range(1, 10):
                json_str = template.format(v=v, a=a)
                target_ids = self.tokenizer(json_str, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
                combinations.append((v, a))
                target_ids_list.append(target_ids)
                
        # log_probs_matrix: (batch_size, 81)
        log_probs_matrix = np.zeros((batch_size, 81))
        
        with torch.no_grad():
            for idx, target_ids in enumerate(target_ids_list):
                # Expand target_ids to batch_size
                target_ids_batch = target_ids.unsqueeze(0).expand(batch_size, -1)
                seq_ids = torch.cat([input_ids, target_ids_batch], dim=1)
                
                target_mask_batch = torch.ones_like(target_ids_batch)
                seq_mask = torch.cat([attention_mask, target_mask_batch], dim=1)
                
                outputs = self.model(input_ids=seq_ids, attention_mask=seq_mask)
                logits = outputs.logits # (batch_size, seq_len + target_len, vocab_size)
                
                input_len = input_ids.shape[1]
                target_len = target_ids.shape[0]
                
                pred_logits = logits[:, input_len - 1 : input_len - 1 + target_len, :]
                
                log_probs_seq = torch.nn.functional.log_softmax(pred_logits, dim=-1)
                
                # Gather log probs for target
                target_ids_expanded = target_ids_batch.unsqueeze(-1)
                target_log_probs = torch.gather(log_probs_seq, dim=-1, index=target_ids_expanded).squeeze(-1)
                
                seq_log_prob = target_log_probs.sum(dim=1).cpu().numpy()
                log_probs_matrix[:, idx] = seq_log_prob
                
        results = []
        for b in range(batch_size):
            log_probs = log_probs_matrix[b]
            max_log_prob = np.max(log_probs)
            exp_probs = np.exp(log_probs - max_log_prob)
            probs = exp_probs / np.sum(exp_probs)
            
            e_v, e_a = 0.0, 0.0
            prob_matrix = np.zeros((9, 9))
            
            for idx, (v, a) in enumerate(combinations):
                p = probs[idx]
                e_v += v * p
                e_a += a * p
                prob_matrix[v-1, a-1] = p
                
            results.append({
                "E_V": float(e_v),
                "E_A": float(e_a),
                "prob_matrix": prob_matrix.tolist(),
                "log_probs": log_probs.tolist()
            })
            
        return results

