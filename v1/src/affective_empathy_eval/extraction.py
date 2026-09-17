import os
import numpy as np
from typing import Dict, List, Optional, Any
from .manifests import ExtractionManifest, ManifestManager

class MockRepresentationExtractor:
    """Mock extractor for dry-run mode and testing without GPU/PyTorch dependencies."""
    def __init__(self, num_layers: int = 12, hidden_dim: int = 768, seed: int = 42, 
                 model_name: str = "mock-model", model_revision: str = "mock-rev"):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)
        self.model_name = model_name
        self.model_revision = model_revision

    def extract_representations(
        self, 
        request_id: str, 
        stimulus_id: str, 
        condition: str, 
        stimulus_text: str,
        full_prompt_text: str,
        output_dir: str,
        manifest_manager: ManifestManager,
        layers_to_extract: Optional[List[int]] = None
    ) -> List[ExtractionManifest]:
        
        text_seed = abs(hash(stimulus_text or stimulus_id or "baseline")) % (2**32)
        local_rng = np.random.default_rng(text_seed)
        
        target_layers = layers_to_extract or list(range(self.num_layers))
        manifests = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for layer in target_layers:
            base_vec = local_rng.normal(loc=0.0, scale=1.0, size=(self.hidden_dim,)).astype(np.float16)
            
            positions = {
                "stimulus_last_token": base_vec + np.float16(layer * 0.1),
                "stimulus_mean_pool": (base_vec * 0.8).astype(np.float16),
                "prompt_last_token": (base_vec * 1.1).astype(np.float16),
                "first_generated_token_input": (base_vec * 1.2).astype(np.float16)
            }
            
            for pos_name, tensor in positions.items():
                tensor_filename = f"{request_id}_L{layer}_{pos_name}.npy"
                tensor_path = os.path.join(output_dir, tensor_filename)
                
                np.save(tensor_path, tensor)
                
                manifest = ExtractionManifest(
                    tensor_path=tensor_path,
                    stimulus_id=stimulus_id,
                    condition=condition,
                    model_name=self.model_name,
                    model_revision=self.model_revision,
                    layer=layer,
                    extraction_position=pos_name,
                    token_offset=-1, # Mock doesn't track exact tokens
                    prompt_hash=str(hash(full_prompt_text)),
                    tensor_shape=list(tensor.shape),
                    dtype="float16"
                )
                manifest_manager.add_manifest(manifest)
                manifests.append(manifest)
                
        manifest_manager.save()
        return manifests

    def extract_representations_batch(
        self,
        requests: List[Dict[str, Any]],
        output_dir: str,
        manifest_manager: ManifestManager,
        layers_to_extract: Optional[List[int]] = None
    ) -> List[ExtractionManifest]:
        all_manifests = []
        for req in requests:
            manifests = self.extract_representations(
                request_id=req["request_id"],
                stimulus_id=req["stimulus_id"],
                condition=req["condition"],
                stimulus_text=req["stimulus_text"],
                full_prompt_text=req["full_prompt_text"],
                output_dir=output_dir,
                manifest_manager=manifest_manager,
                layers_to_extract=layers_to_extract
            )
            all_manifests.extend(manifests)
        return all_manifests


class PyTorchRepresentationExtractor:
    """
    Extracts representations from PyTorch/HuggingFace Transformer models using forward hooks.
    Phase B1: Extracts hidden states only at specified positions.
    """
    def __init__(self, model: Any, tokenizer: Any, model_name: str, model_revision: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_revision = model_revision
        self.activations: Dict[int, Any] = {}
        self.hooks = []

    def _register_hooks(self, target_layers: List[int]):
        self.activations.clear()
        
        transformer_layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers # Llama/Qwen
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            transformer_layers = self.model.transformer.h # GPT/Qwen older
            
        if transformer_layers is None:
            raise ValueError("Could not find transformer layers in the model architecture.")

        for idx in target_layers:
            if idx < len(transformer_layers):
                def get_hook(layer_idx):
                    def hook(module, input, output):
                        # output[0] is usually the hidden states tuple
                        tensor = output[0] if isinstance(output, tuple) else output
                        # Store in dict on CPU to avoid GPU OOM, cast to float16
                        self.activations[layer_idx] = tensor.detach().cpu().numpy().astype(np.float16)
                    return hook
                
                h = transformer_layers[idx].register_forward_hook(get_hook(idx))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        
    def _find_stimulus_token_offsets(self, full_input_ids: np.ndarray, stimulus_text: str) -> Optional[tuple[int, int]]:
        """
        Heuristic to find the start and end offsets of the stimulus within the prompt.
        For rigorous use, the chat template generation should provide the exact offsets.
        """
        # This is a fallback implementation if offsets are not provided. 
        # In a real rigorous setting, you would pass exact token offsets from the tokenizer/template.
        stim_ids = self.tokenizer.encode(stimulus_text, add_special_tokens=False)
        if len(stim_ids) == 0:
            return None
            
        full_ids_list = full_input_ids.tolist()
        stim_len = len(stim_ids)
        
        # Simple sublist search
        for i in range(len(full_ids_list) - stim_len + 1):
            if full_ids_list[i:i+stim_len] == stim_ids:
                return (i, i + stim_len - 1)
        
        # If not found (due to tokenization boundary changes), fallback to last tokens
        return None

    def extract_representations(
        self, 
        request_id: str, 
        stimulus_id: str, 
        condition: str, 
        stimulus_text: str,
        full_prompt_text: str,
        output_dir: str,
        manifest_manager: ManifestManager,
        layers_to_extract: Optional[List[int]] = None,
        stimulus_offset_start: Optional[int] = None,
        stimulus_offset_end: Optional[int] = None
    ) -> List[ExtractionManifest]:
        
        if layers_to_extract is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                layers_to_extract = list(range(self.model.config.num_hidden_layers))
            else:
                raise ValueError("Must provide layers_to_extract if model config has no num_hidden_layers.")
                
        self._register_hooks(layers_to_extract)
        os.makedirs(output_dir, exist_ok=True)
        manifests = []
        prompt_hash = str(hash(full_prompt_text))
        
        try:
            inputs = self.tokenizer(full_prompt_text, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            import torch
            with torch.no_grad():
                _ = self.model(**inputs)
                
            input_ids = inputs["input_ids"][0].cpu().numpy()
            prompt_last_idx = len(input_ids) - 1
            first_generated_idx = prompt_last_idx # Usually context window end right before generation
            
            # Identify stimulus boundaries
            if stimulus_offset_start is None or stimulus_offset_end is None:
                offsets = self._find_stimulus_token_offsets(input_ids, stimulus_text)
                if offsets:
                    stimulus_offset_start, stimulus_offset_end = offsets
                else:
                    # Fallback to prompt last if stimulus text can't be identified
                    stimulus_offset_start = prompt_last_idx
                    stimulus_offset_end = prompt_last_idx
            
            for layer_idx, act in self.activations.items():
                # act shape: (1, seq_len, hidden_dim)
                
                # 1. prompt_last_token
                prompt_last_tensor = act[0, prompt_last_idx, :]
                
                # 2. stimulus_last_token
                stimulus_last_tensor = act[0, stimulus_offset_end, :]
                
                # 3. stimulus_mean_pool
                if stimulus_offset_start <= stimulus_offset_end:
                    stimulus_mean_tensor = np.mean(act[0, stimulus_offset_start:stimulus_offset_end+1, :], axis=0).astype(np.float16)
                else:
                    stimulus_mean_tensor = stimulus_last_tensor
                    
                # 4. first_generated_token_input (in standard causal LM, this is the state at prompt_last_idx)
                first_gen_tensor = act[0, first_generated_idx, :]
                
                positions = {
                    "prompt_last_token": (prompt_last_tensor, prompt_last_idx),
                    "stimulus_last_token": (stimulus_last_tensor, stimulus_offset_end),
                    "stimulus_mean_pool": (stimulus_mean_tensor, -1),
                    "first_generated_token_input": (first_gen_tensor, first_generated_idx)
                }
                
                for pos_name, (tensor, offset) in positions.items():
                    tensor_filename = f"{request_id}_L{layer_idx}_{pos_name}.npy"
                    tensor_path = os.path.join(output_dir, tensor_filename)
                    
                    np.save(tensor_path, tensor)
                    
                    manifest = ExtractionManifest(
                        tensor_path=tensor_path,
                        stimulus_id=stimulus_id,
                        condition=condition,
                        model_name=self.model_name,
                        model_revision=self.model_revision,
                        layer=layer_idx,
                        extraction_position=pos_name,
                        token_offset=offset,
                        prompt_hash=prompt_hash,
                        tensor_shape=list(tensor.shape),
                        dtype="float16"
                    )
                    manifest_manager.add_manifest(manifest)
                    manifests.append(manifest)
                    
            manifest_manager.save()
            return manifests
            
        finally:
            self.remove_hooks()

    def extract_representations_batch(
        self,
        requests: List[Dict[str, Any]],
        output_dir: str,
        manifest_manager: ManifestManager,
        layers_to_extract: Optional[List[int]] = None
    ) -> List[ExtractionManifest]:
        
        if not requests:
            return []
            
        if layers_to_extract is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                layers_to_extract = list(range(self.model.config.num_hidden_layers))
            else:
                raise ValueError("Must provide layers_to_extract if model config has no num_hidden_layers.")
                
        self._register_hooks(layers_to_extract)
        os.makedirs(output_dir, exist_ok=True)
        manifests = []
        
        # Setup tokenizer for left padding
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        full_prompts = [req["full_prompt_text"] for req in requests]
        
        try:
            inputs = self.tokenizer(full_prompts, padding=True, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            import torch
            with torch.no_grad():
                _ = self.model(**inputs)
                
            input_ids_np = inputs["input_ids"].cpu().numpy()
            attention_mask_np = inputs["attention_mask"].cpu().numpy()
            
            for i, req in enumerate(requests):
                seq_len = input_ids_np.shape[1]
                prompt_last_idx = seq_len - 1
                first_generated_idx = prompt_last_idx
                prompt_hash = str(hash(req["full_prompt_text"]))
                
                stimulus_offset_start = req.get("stimulus_offset_start")
                stimulus_offset_end = req.get("stimulus_offset_end")
                
                if stimulus_offset_start is None or stimulus_offset_end is None:
                    # Search within this specific sequence
                    offsets = self._find_stimulus_token_offsets(input_ids_np[i], req["stimulus_text"])
                    if offsets:
                        stimulus_offset_start, stimulus_offset_end = offsets
                    else:
                        stimulus_offset_start = prompt_last_idx
                        stimulus_offset_end = prompt_last_idx
                
                for layer_idx, act in self.activations.items():
                    # act shape: (batch_size, seq_len, hidden_dim)
                    prompt_last_tensor = act[i, prompt_last_idx, :]
                    stimulus_last_tensor = act[i, stimulus_offset_end, :]
                    
                    if stimulus_offset_start <= stimulus_offset_end:
                        stimulus_mean_tensor = np.mean(act[i, stimulus_offset_start:stimulus_offset_end+1, :], axis=0).astype(np.float16)
                    else:
                        stimulus_mean_tensor = stimulus_last_tensor
                        
                    first_gen_tensor = act[i, first_generated_idx, :]
                    
                    positions = {
                        "prompt_last_token": (prompt_last_tensor, prompt_last_idx),
                        "stimulus_last_token": (stimulus_last_tensor, stimulus_offset_end),
                        "stimulus_mean_pool": (stimulus_mean_tensor, -1),
                        "first_generated_token_input": (first_gen_tensor, first_generated_idx)
                    }
                    
                    for pos_name, (tensor, offset) in positions.items():
                        tensor_filename = f"{req['request_id']}_L{layer_idx}_{pos_name}.npy"
                        tensor_path = os.path.join(output_dir, tensor_filename)
                        
                        np.save(tensor_path, tensor)
                        
                        manifest = ExtractionManifest(
                            tensor_path=tensor_path,
                            stimulus_id=req["stimulus_id"],
                            condition=req["condition"],
                            model_name=self.model_name,
                            model_revision=self.model_revision,
                            layer=layer_idx,
                            extraction_position=pos_name,
                            token_offset=offset,
                            prompt_hash=prompt_hash,
                            tensor_shape=list(tensor.shape),
                            dtype="float16"
                        )
                        manifest_manager.add_manifest(manifest)
                        manifests.append(manifest)
                        
            manifest_manager.save()
            return manifests
            
        finally:
            self.tokenizer.padding_side = original_padding_side
            self.remove_hooks()
