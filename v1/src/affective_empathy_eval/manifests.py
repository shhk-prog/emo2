import json
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class ExtractionManifest:
    """
    Manifest for a single extracted activation tensor.
    Keeps track of all necessary metadata to ensure reproducibility and correct alignment.
    """
    tensor_path: str                 # Relative path to the saved .npy or .npz file
    stimulus_id: str                 # Unique ID of the stimulus
    condition: str                   # Extraction condition (e.g., 'empty_baseline', 'post_reported_va')
    model_name: str                  # Name of the model
    model_revision: str              # Model revision / commit hash
    layer: int                       # Layer number from which the activation was extracted
    extraction_position: str         # The position identifier (e.g., 'stimulus_last_token')
    token_offset: Optional[int]      # Actual absolute index in the sequence (if applicable)
    prompt_hash: str                 # Hash of the exact prompt used
    tensor_shape: List[int]          # Shape of the extracted tensor
    dtype: str                       # Data type (e.g., 'float16')

class ManifestManager:
    """
    Manages the saving and loading of extraction manifests.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.manifests: List[ExtractionManifest] = []
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.output_dir, "activation_manifest.jsonl")

    def add_manifest(self, manifest: ExtractionManifest):
        self.manifests.append(manifest)
        
    def save(self):
        """Append all currently held manifests to the JSONL file and clear the buffer."""
        if not self.manifests:
            return
            
        with open(self.manifest_path, "a") as f:
            for manifest in self.manifests:
                f.write(json.dumps(asdict(manifest)) + "\n")
        
        self.manifests = [] # clear after saving
        
    def load_all_as_dataframe(self) -> pd.DataFrame:
        """Loads the entire manifest file into a pandas DataFrame."""
        if not os.path.exists(self.manifest_path):
            return pd.DataFrame()
            
        records = []
        with open(self.manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)
