#!/usr/bin/env python3
"""Accurately simulate ICLR 2027 page layout height for main text of iclr2027_conference2.tex.

ICLR style specifications (iclr2027_conference.sty):
- \textheight = 9.0 in = 648 pt
- \textwidth = 5.5 in = 396 pt
- Main font: 10pt on 12pt baselineskip (\normalsize: 10pt/12pt)
- Title + Authors + Margin: approx 180-220 pt
- Abstract: approx 140 pt
- Section heading: \section = ~24 pt (12pt space before + 10pt font + 6pt space after)
- Subsection heading: \subsection = ~18 pt
- Table floats [t]:
  - Table 1 (tab:main-stage-overview): ~150 pt
  - Table 2 (tab:main-key-results): ~160 pt
  - Table 3 (tab:main-v3-summary): ~170 pt
- Display math \[ ... \]: ~24-36 pt each
- Normal text line: 12 pt
"""

import re
from pathlib import Path

tex_path = Path("/mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference2.tex")
content = tex_path.read_text(encoding="utf-8")

match = re.search(r"\\maketitle\s+(.*?)\\bibliography\{", content, re.DOTALL)
main_text = match.group(1)

# ICLR constants in pt
PAGE_HEIGHT = 648.0 # 9 inches

# Title block (title, authors, institutions)
# Title is 14pt bold, authors 10pt
title_height = 200.0

# Abstract
abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_text, re.DOTALL)
abstract_text = abstract_match.group(1).strip()
# Abstract font is 9pt/11pt, textwidth narrower (4.5 in = ~324 pt)
# Japanese chars per line: ~324 pt / 9 pt ~= 36 chars/line
ab_jp_chars = len(re.findall(r"[\u3000-\u9faf]", abstract_text))
ab_en_words = len(re.findall(r"[a-zA-Z0-9_-]+", abstract_text))
ab_eq_chars = ab_jp_chars + ab_en_words * 2
ab_lines = ab_eq_chars / 36.0 + 4 # equation in abstract
abstract_height = 30.0 + ab_lines * 11.0 + 20.0 # heading + lines + space after

# Remove abstract from main_text for body analysis
body_text = main_text.replace(abstract_match.group(0), "")

# Extract tables
tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", body_text, re.DOTALL)
table_heights = []
for i, t in enumerate(tables):
    # Count rows in tabular
    rows = t.count(r"\\")
    # approx caption (30pt) + rows (14pt each) + rules (10pt) + float margins (20pt)
    h = 30.0 + rows * 14.0 + 30.0
    table_heights.append(h)

for t in tables:
    body_text = body_text.replace(t, "")

# Count sections, subsections, paragraphs
n_sections = len(re.findall(r"\\section\{", body_text))
n_subsections = len(re.findall(r"\\subsection\{", body_text))
n_paragraphs = len(re.findall(r"\\paragraph\{", body_text))

# Extract display math
display_math = re.findall(r"\\\[.*?\\\]", body_text, re.DOTALL)
display_math_height = 0.0
for m in display_math:
    lines_in_m = max(1, m.count(r"\\") + m.count("\n") - 1)
    display_math_height += 16.0 + lines_in_m * 14.0

for m in display_math:
    body_text = body_text.replace(m, "")

# Clean body text
clean_body = re.sub(r"%.*?\n", "\n", body_text)
clean_body = re.sub(r"\\[a-zA-Z]+\*?(?:\[.*?\])?(?:\{.*?\})?", " ", clean_body)
clean_body = re.sub(r"\s+", " ", clean_body).strip()

jp_chars = len(re.findall(r"[\u3000-\u9faf]", clean_body))
en_words = len(re.findall(r"[a-zA-Z0-9_-]+", clean_body))
eq_chars = jp_chars + en_words * 2

# Textwidth is 396 pt. In 10pt Japanese, approx 39-40 full-width chars per line.
chars_per_line = 39.5
text_lines = eq_chars / chars_per_line
text_height = text_lines * 12.0 # 12pt baselineskip

headings_height = n_sections * 24.0 + n_subsections * 18.0 + n_paragraphs * 14.0
total_tables_height = sum(table_heights)

total_height = (title_height + abstract_height + headings_height + 
                text_height + display_math_height + total_tables_height)

total_pages = total_height / PAGE_HEIGHT

print(f"--- ICLR 2027 Main Text Height Simulation ---")
print(f"Title Block: {title_height:.1f} pt")
print(f"Abstract: {abstract_height:.1f} pt ({ab_eq_chars} eq-chars, ~{ab_lines:.1f} lines)")
print(f"Headings ({n_sections} sec, {n_subsections} subsec, {n_paragraphs} para): {headings_height:.1f} pt")
print(f"Body Text: {text_height:.1f} pt ({eq_chars} eq-chars, ~{text_lines:.1f} lines)")
print(f"Display Math ({len(display_math)} eqns): {display_math_height:.1f} pt")
print(f"Tables ({len(tables)} tables): {total_tables_height:.1f} pt (T1: {table_heights[0]:.1f}, T2: {table_heights[1]:.1f}, T3: {table_heights[2]:.1f})")
print(f"--------------------------------------------------")
print(f"Total Height: {total_height:.1f} pt")
print(f"Page Height: {PAGE_HEIGHT:.1f} pt")
print(f"Estimated Total Pages: {total_pages:.2f} pages")
print(f"Target: <= 9.0 pages")
if total_pages <= 9.0:
    print(f"RESULT: PASS (Leaves {9.0 - total_pages:.2f} pages margin)")
else:
    print(f"RESULT: OVERFLOW by {total_pages - 9.0:.2f} pages")
