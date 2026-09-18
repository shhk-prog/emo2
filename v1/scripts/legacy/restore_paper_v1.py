#!/usr/bin/env python3
"""
Restore paper_v1.md from Git HEAD and re-append clean Phase A/B/C sections.
"""

import subprocess
import sys
import os

def main():
    target_path = "v3/docs/paper_v1.md"
    
    print("Checking Git status for clean version of paper_v1.md...")
    try:
        # Try getting the committed version from git HEAD
        result = subprocess.run(
            ["git", "show", "HEAD:v3/docs/paper_v1.md"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        clean_text = result.stdout.decode("utf-8")
        print("Successfully retrieved clean base from Git HEAD!")
    except Exception as e:
        print(f"Failed to retrieve from HEAD directly: {e}")
        # Try git checkout
        print("Attempting to restore using git checkout...")
        subprocess.run(["git", "checkout", "HEAD", "--", target_path], check=True)
        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            clean_text = f.read()

    # Load clean Phase B comprehensive report
    phase_b_report_path = "v1/results/derived/v1_phase_b/phase_b_comprehensive_report.md"
    phase_b_content = ""
    if os.path.exists(phase_b_report_path):
        with open(phase_b_report_path, "r", encoding="utf-8") as f:
            phase_b_content = f.read()

    # If the clean_text has trailing headers, find appropriate append point
    cut_marker = "# phase Aの結果"
    if cut_marker in clean_text:
        base_clean = clean_text.split(cut_marker)[0].strip()
    else:
        # In case the file ended earlier
        base_clean = clean_text.strip()

    # Construct the restored document
    restored_sections = [
        base_clean,
        "",
        "# phase Aの結果",
        "",
        "# phase Aの考察",
        "",
        "",
        "# phase Bの結果",
        phase_b_content,
        "",
        "# phase Bの考察",
        "",
        "",
        "# phase Cの結果",
        "",
        "# phase Cの考察",
        ""
    ]

    final_content = "\n".join(restored_sections)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(final_content)

    print(f"Successfully restored and reconstructed: {target_path}")
    print("Verification: First 5 lines:")
    for line in final_content.splitlines()[:5]:
        print("  ", line)

if __name__ == "__main__":
    main()
