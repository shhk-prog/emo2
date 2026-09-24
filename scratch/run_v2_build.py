#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from v2.scripts.build_paper_summary import build_v2_summary
from scripts.paper_summary_manifest import PaperSummaryManifest

manifest = PaperSummaryManifest()
v2_dir = root / "v2"
out_dir = root / "results" / "derived" / "paper_summary"

print("Building V2 summary...")
df, qc = build_v2_summary(v2_dir, out_dir, manifest, strict=False)
print("V2 build complete. QC info:", qc)
