#!/usr/bin/env python3
"""Analyze main text of iclr2027_conference2.tex for length, page count estimation, and factual consistency."""
import re
from pathlib import Path

tex_path = Path("/mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference2.tex")
content = tex_path.read_text(encoding="utf-8")

# Extract main text: between \maketitle and \bibliography
pattern = r"\\maketitle\s+(.*?)\\bibliography\{"
match = re.search(pattern, content, re.DOTALL)
if not match:
    print("Could not find main text boundaries!")
    exit(1)

main_text = match.group(1)

lines = main_text.splitlines()
total_lines = len(lines)
non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith("%")]

# Separate tables
tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", main_text, re.DOTALL)
display_math = re.findall(r"\\\[.*?\\\]", main_text, re.DOTALL)

# Strip comments, tables, math from text
text_clean = main_text
for t in tables:
    text_clean = text_clean.replace(t, " ")
for m in display_math:
    text_clean = text_clean.replace(m, " ")
text_clean = re.sub(r"%.*?\n", "\n", text_clean)
text_clean = re.sub(r"\\[a-zA-Z]+\*?(?:\[.*?\])?(?:\{.*?\})?", " ", text_clean)
text_clean = re.sub(r"\s+", " ", text_clean).strip()

# Japanese character count + English word count
jp_chars = len(re.findall(r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\u4e00-\u9faf]", text_clean))
en_words = len(re.findall(r"[a-zA-Z0-9_-]+", text_clean))

print(f"Total lines in Main: {total_lines}")
print(f"Non-empty lines: {len(non_empty_lines)}")
print(f"Tables: {len(tables)}")
print(f"Display math equations: {len(display_math)}")
print(f"Japanese characters: {jp_chars}")
print(f"English words: {en_words}")
print(f"Estimated equivalent Japanese characters: {jp_chars + en_words * 2}")
