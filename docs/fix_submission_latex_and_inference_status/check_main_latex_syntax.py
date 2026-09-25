#!/usr/bin/env python3
"""Thoroughly check main text for potential LaTeX issues, typos, and formatting errors."""
import re
from pathlib import Path

tex_path = Path("/mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference2.tex")
content = tex_path.read_text(encoding="utf-8")

match = re.search(r"\\maketitle\s+(.*?)\\bibliography\{", content, re.DOTALL)
main_text = match.group(1)

# Check unmatched brackets or environments
env_begin = re.findall(r"\\begin\{([a-zA-Z*]+)\}", main_text)
env_end = re.findall(r"\\end\{([a-zA-Z*]+)\}", main_text)
print(f"Environments begin: {env_begin}")
print(f"Environments end:   {env_end}")
if env_begin == env_end:
    print("All environments match perfectly!")
else:
    print("MISMATCH IN ENVIRONMENTS!")

# Check display math \[ and \]
dm_open = main_text.count(r"\[")
dm_close = main_text.count(r"\]")
print(f"Display math open: {dm_open}, close: {dm_close}")
if dm_open == dm_close:
    print("Display math delimiters match!")
else:
    print("MISMATCH IN DISPLAY MATH!")

# Check inline math $ ... $
dollar_count = main_text.count("$")
print(f"Dollar count: {dollar_count} (even: {dollar_count % 2 == 0})")

# Check curly braces balance
open_braces = main_text.count("{")
close_braces = main_text.count("}")
print(f"Open braces: {open_braces}, Close braces: {close_braces}")
if open_braces == close_braces:
    print("Curly braces match perfectly!")
else:
    print("MISMATCH IN CURLY BRACES!")

# Check for undefined or weird commands
commands = re.findall(r"\\([a-zA-Z]+)", main_text)
from collections import Counter
cmd_counts = Counter(commands)
print("\nUnique commands used in Main:")
for cmd, c in sorted(cmd_counts.items()):
    print(f"  \\{cmd}: {c}")
