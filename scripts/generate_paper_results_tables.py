#!/usr/bin/env python3
"""
Master Orchestrator for Generating ICLR 2027 Paper Results Tables.

This script executes all stage-specific summarization modules:
1. Behavioral (EmoBank): 3-Way (Writer vs Reader vs Self) VAD Correspondence & Cognitive Coupling
2. Behavioral (AIPsy): Emotional Sensitivity (RQ1) & Internal Coupling (RQ4)
3. V1 (Representation Sharing): E1 Peak Decodability & E3 Causal Map (Reader vs Self)
4. V2 (Post-training Reorganization): H1-H2 Geometry/Sharing & H3 Causal Relocation LMM (Base vs Instruct)
5. V3 (Causal Utilization): Gate Evaluation, Spatiotemporal Dissociation, & Confirmatory Matrix

Usage:
    python3 scripts/generate_paper_results_tables.py [--repo-root .] [--out-dir iclr2027/tables]
"""

import os
import sys
import argparse

# Add repo root and scripts directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import summarize_behavioral_emobank
import summarize_behavioral_aipsy
import summarize_v1_internal_sharing
import summarize_v2_reorganization
import summarize_v3_causal_utilization

def run_all(repo_root, out_dir, stages=None):
    if stages is None:
        stages = ["behavioral", "v1", "v2", "v3"]
    stages = [s.lower() for s in stages]

    print("=" * 70)
    print("Generating Publication-Ready Tables for ICLR 2027 Paper")
    print(f"Repository Root: {repo_root}")
    print(f"Output Directory: {out_dir}")
    print(f"Target Stages: {stages}")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)

    # 1 & 2. Behavioral
    if "behavioral" in stages or "behavioral_emobank" in stages:
        print("\n[1] Processing Behavioral Stage: EmoBank 3-Way VAD Correspondence...")
        sys.argv = ["summarize_behavioral_emobank.py", "--repo-root", repo_root, "--out-dir", out_dir]
        summarize_behavioral_emobank.main()

    if "behavioral" in stages or "behavioral_aipsy" in stages:
        print("\n[2] Processing Behavioral Stage: AIPsy Sensitivity & Coupling...")
        sys.argv = ["summarize_behavioral_aipsy.py", "--repo-root", repo_root, "--out-dir", out_dir]
        summarize_behavioral_aipsy.main()

    # 3. V1 (Internal Representation Sharing)
    if "v1" in stages:
        print("\n[3] Processing V1 Stage: Internal Representation Sharing (E1-E6)...")
        sys.argv = ["summarize_v1_internal_sharing.py", "--repo-root", repo_root, "--out-dir", out_dir]
        summarize_v1_internal_sharing.main()

    # 4. V2 (Post-training Reorganization)
    if "v2" in stages:
        print("\n[4] Processing V2 Stage: Post-training-Associated Reorganization (H1-H4)...")
        sys.argv = ["summarize_v2_reorganization.py", "--repo-root", repo_root, "--out-dir", out_dir]
        summarize_v2_reorganization.main()

    # 5. V3 (Causal Utilization & Spatiotemporal Dynamics)
    if "v3" in stages:
        print("\n[5] Processing V3 Stage: Causal Utilization and Spatiotemporal Dynamics...")
        sys.argv = ["summarize_v3_causal_utilization.py", "--repo-root", repo_root, "--out-dir", out_dir]
        summarize_v3_causal_utilization.main()

    print("\n" + "=" * 70)
    print("Requested tables successfully generated!")
    print(f"Location: {os.path.abspath(out_dir)}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Master Table Generator for ICLR 2027 Paper")
    parser.add_argument("--repo-root", default=REPO_ROOT, help="Root directory of the repo")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "iclr2027/tables"), help="Output directory for LaTeX tables")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["behavioral", "v1", "v2", "v3"],
        help="Stages to generate (e.g. --stages behavioral v1 v3)",
    )
    args = parser.parse_args()

    run_all(args.repo_root, args.out_dir, stages=args.stages)

if __name__ == "__main__":
    main()
