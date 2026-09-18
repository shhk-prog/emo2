import torch
import numpy as np

def generate_81_candidates(template_type="standard"):
    """
    Generates 81 candidate strings for V and A ranging from 1 to 9.
    template_type: 
        "standard": {"valence": V, "arousal": A}
        "reversed": {"arousal": A, "valence": V}
    """
    candidates = []
    va_pairs = []
    for v in range(1, 10):
        for a in range(1, 10):
            if template_type == "standard":
                s = f'{{"valence": {v}, "arousal": {a}}}'
            elif template_type == "reversed":
                s = f'{{"arousal": {a}, "valence": {v}}}'
            else:
                raise ValueError(f"Unknown template_type: {template_type}")
            candidates.append(s)
            va_pairs.append((v, a))
    return candidates, va_pairs

def compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates, device="cuda", normalize_length=False):
    """
    Given a prompt (e.g. chat history up to the point of assistant response),
    computes the log likelihood (or length-normalized log likelihood) of each candidate string.
    
    Note:
      - Raw sequence log-likelihood: sum_{t} log P(token_t | prompt + candidate_{<t})
      - Length-normalized log-likelihood: (1 / |candidate|) * sum_{t} log P(token_t | prompt + candidate_{<t})
      
    Returns:
      likelihoods: List[float] (the computed log-likelihood score for each candidate)
      candidate_lengths: List[int] (the number of tokens in each candidate string)
      *IMPORTANT*: candidate_lengths is NOT a probability distribution.
      To obtain normalized probabilities over candidates, pass likelihoods to `compute_expected_va(...)`.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    likelihoods = []
    candidate_lengths = []
    
    # We can batch this or do it in a simple loop. Loop is safer for variable length candidates.
    for cand in candidates:
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        input_ids = prompt_ids + cand_ids
        input_tensor = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits[0, :-1, :] # (seq_len-1, vocab_size)
            
        # We want the log probs of cand_ids
        # The first token of cand_ids is predicted by the last token of prompt_ids
        prompt_end_idx = len(prompt_ids) - 1
        cand_log_probs = []
        for i, token_id in enumerate(cand_ids):
            logit_step = logits[prompt_end_idx + i]
            log_probs = torch.nn.functional.log_softmax(logit_step, dim=-1)
            cand_log_probs.append(log_probs[token_id].item())
            
        raw_ll = sum(cand_log_probs)
        cand_len = len(cand_ids)
        if normalize_length and cand_len > 0:
            likelihood = raw_ll / cand_len
        else:
            likelihood = raw_ll
            
        likelihoods.append(likelihood)
        candidate_lengths.append(cand_len)
        
    return likelihoods, candidate_lengths

def compute_expected_va(likelihoods, va_pairs, tau=1.0):
    """
    Given log likelihoods, computes the probability distribution, E[V], E[A], entropy, and p(5,5).
    """
    # log-sum-exp trick
    l_arr = np.array(likelihoods) / tau
    l_max = np.max(l_arr)
    probs = np.exp(l_arr - l_max)
    probs = probs / np.sum(probs)
    
    E_v = 0.0
    E_a = 0.0
    p_55 = 0.0
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    
    for p, (v, a) in zip(probs, va_pairs):
        E_v += p * v
        E_a += p * a
        if v == 5 and a == 5:
            p_55 = p
            
    return E_v, E_a, entropy, p_55, probs
