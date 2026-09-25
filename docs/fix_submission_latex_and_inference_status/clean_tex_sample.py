import re

with open("iclr2027/iclr2027_conference2.tex", "r", encoding="utf-8") as f:
    content = f.read()

# Pattern to match from \end{abstract} through the sample main text,
# sample bibliography, and sample \appendix \section{Appendix}...
pattern = r"\\end\{abstract\}\s*\\section\{Submission of conference papers to ICLR 2027\}.*?\\section\{Appendix\}\s*You may include other additional sections here\."

match = re.search(pattern, content, flags=re.DOTALL)
if match:
    print(f"Matched {len(match.group(0))} characters")
    replacement = r"""\end{abstract}

\bibliography{iclr2027_conference}
\bibliographystyle{iclr2027_conference}

\appendix"""
    new_content = content[:match.start()] + replacement + content[match.end():]
    with open("iclr2027/iclr2027_conference2.tex", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully replaced sample text in iclr2027_conference2.tex")
else:
    print("Pattern not matched!")
