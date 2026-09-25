#!/usr/bin/env python3
"""Validate citations and labels in the main text of iclr2027_conference2.tex."""
import re
from pathlib import Path

tex_path = Path("/mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference2.tex")
bib_path = Path("/mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference.bib")

tex_content = tex_path.read_text(encoding="utf-8")
bib_content = bib_path.read_text(encoding="utf-8")

# Extract main text
match = re.search(r"\\maketitle\s+(.*?)\\bibliography\{", tex_content, re.DOTALL)
main_text = match.group(1)

# Find all citations in main text
citations = set()
for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", main_text):
    for key in m.group(1).split(","):
        citations.add(key.strip())

# Find all defined bib keys
bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib_content))

missing_cites = [c for c in sorted(citations) if c not in bib_keys]
print(f"Total unique citations in Main text: {len(citations)}")
if missing_cites:
    print(f"MISSING CITATIONS in bib: {missing_cites}")
else:
    print("All citations in Main text exist in bib file!")

# Find all \ref and \pageref in main text
refs = set()
for m in re.finditer(r"\\(?:page)?ref\{([^}]+)\}", main_text):
    refs.add(m.group(1).strip())

# Find all \label in the entire tex file
all_labels = set(re.findall(r"\\label\{([^}]+)\}", tex_content))

missing_refs = [r for r in sorted(refs) if r not in all_labels]
print(f"Total unique \\ref in Main text: {len(refs)}")
if missing_refs:
    print(f"MISSING LABELS in entire tex file: {missing_refs}")
else:
    print("All \\ref in Main text are properly defined in the document!")

print("\n--- Detailed refs in Main text ---")
for r in sorted(refs):
    defined_in_main = r in re.findall(r"\\label\{([^}]+)\}", main_text)
    print(f"  {r}: {'[MAIN]' if defined_in_main else '[APPENDIX]'}")
